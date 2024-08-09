from playwright.async_api import Page, CDPSession, ViewportSize
from typing import Any, TypedDict
import re
from .utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
    DOMNode,
)

IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)


IN_VIEWPORT_RATIO_THRESHOLD = 0.6

CURRENT_VIEWPORT_ONLY = True


async def fetch_browser_info(
    page: Page,
    client: CDPSession,
) -> BrowserInfo:
    viewport_size = page.viewport_size
    assert viewport_size is not None, "Viewport size is None"

    # extract domtree
    tree = await client.send(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    # calibrate the bounds, in some cases, the bounds are scaled somehow
    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / viewport_size["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    # extract browser info
    win_top_bound = await page.evaluate("window.pageYOffset")
    win_left_bound = await page.evaluate("window.pageXOffset")
    win_width = await page.evaluate("window.screen.width")
    win_height = await page.evaluate("window.screen.height")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = await page.evaluate("window.devicePixelRatio")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }

    # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
    info: BrowserInfo = {"DOMTree": tree, "config": config}

    return info


async def get_bounding_client_rect(
    client: CDPSession, backend_node_id: str
) -> dict[str, Any]:
    try:
        remote_object = await client.send(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = await client.send(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except Exception:
        return {"result": {"subtype": "error"}}


def get_element_in_viewport_ratio(
    elem_left_bound: float,
    elem_top_bound: float,
    width: float,
    height: float,
    config: BrowserConfig,
) -> float:
    elem_right_bound = elem_left_bound + width
    elem_lower_bound = elem_top_bound + height

    win_left_bound = 0
    win_right_bound = config["win_width"]
    win_top_bound = 0
    win_lower_bound = config["win_height"]

    # Compute the overlap in x and y axes
    overlap_width = max(
        0,
        min(elem_right_bound, win_right_bound) - max(elem_left_bound, win_left_bound),
    )
    overlap_height = max(
        0,
        min(elem_lower_bound, win_lower_bound) - max(elem_top_bound, win_top_bound),
    )

    # Compute the overlap area
    ratio = overlap_width * overlap_height / width * height
    return ratio


async def fetch_page_accessibility_tree(
    info: BrowserInfo,
    client: CDPSession,
    current_viewport_only: bool,
) -> AccessibilityTree:
    accessibility_tree: AccessibilityTree = (
        await client.send("Accessibility.getFullAXTree", {})
    )["nodes"]

    # a few nodes are repeated in the accessibility tree
    seen_ids = set()
    _accessibility_tree = []
    for node in accessibility_tree:
        if node["nodeId"] not in seen_ids:
            _accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    accessibility_tree = _accessibility_tree

    nodeid_to_cursor = {}
    for cursor, node in enumerate(accessibility_tree):
        nodeid_to_cursor[node["nodeId"]] = cursor
        # usually because the node is not visible etc
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
            continue
        backend_node_id = str(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            # always inside the viewport
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        else:
            response = await get_bounding_client_rect(client, backend_node_id)
            if response.get("result", {}).get("subtype", "") == "error":
                node["union_bound"] = None
            else:
                x = response["result"]["value"]["x"]
                y = response["result"]["value"]["y"]
                width = response["result"]["value"]["width"]
                height = response["result"]["value"]["height"]
                node["union_bound"] = [x, y, width, height]

    # filter nodes that are not in the current viewport
    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            # update the node information in the accessibility tree
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
            # update the children of the parent node
            assert accessibility_tree[parent_cursor].get("parentId", "Root") is not None
            # remove the nodeid from parent's childIds
            index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
            accessibility_tree[parent_cursor]["childIds"].pop(index)
            # Insert children_nodeids in the same location
            for child_nodeid in children_nodeids:
                accessibility_tree[parent_cursor]["childIds"].insert(
                    index, child_nodeid
                )
                index += 1
            # update children node's parent
            for child_nodeid in children_nodeids:
                child_cursor = nodeid_to_cursor[child_nodeid]
                accessibility_tree[child_cursor]["parentId"] = parent_nodeid
            # mark as removed
            accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            # invisible node
            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = get_element_in_viewport_ratio(
                elem_left_bound=float(x),
                elem_top_bound=float(y),
                width=float(width),
                height=float(height),
                config=config,
            )

            if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                remove_node_in_graph(node)

        accessibility_tree = [
            node
            for node in accessibility_tree
            if node.get("parentId", "Root") != "[REMOVED]"
        ]

    return accessibility_tree


class ObsNode(TypedDict):
    backend_id: str
    union_bound: list[float] | None
    text: str


ObsNodesInfo = dict[str, ObsNode]


def parse_accessibility_tree(
    accessibility_tree: AccessibilityTree,
) -> tuple[str, ObsNodesInfo]:
    """Parse the accessibility tree into a string text"""
    node_id_to_idx: dict[str, int] = {}
    for idx, node in enumerate(accessibility_tree):
        node_id_to_idx[node["nodeId"]] = idx

    obs_nodes_info: ObsNodesInfo = {}

    def dfs(idx: int, obs_node_id: str, depth: int) -> str:
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass

            if properties:
                node_str += " " + " ".join(properties)

            # check valid
            if not node_str.strip():
                valid_node = False

            # empty generic node
            if not name.strip():
                if not properties:
                    if role in [
                        "generic",
                        "img",
                        "list",
                        "strong",
                        "paragraph",
                        "banner",
                        "navigation",
                        "Section",
                        "LabelText",
                        "Legend",
                        "listitem",
                    ]:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            if valid_node:
                tree_str += f"{indent}{node_str}"
                obs_nodes_info[obs_node_id] = {
                    "backend_id": node["backendDOMNodeId"],
                    "union_bound": node["union_bound"],
                    "text": node_str,
                }

        except Exception:
            valid_node = False

        for _, child_node_id in enumerate(node["childIds"]):
            if child_node_id not in node_id_to_idx:
                continue
            # mark this to save some tokens
            child_depth = depth + 1 if valid_node else depth
            child_str = dfs(node_id_to_idx[child_node_id], child_node_id, child_depth)
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
    return tree_str, obs_nodes_info


def clean_accesibility_tree(tree_str: str) -> str:
    """further clean accesibility tree"""
    clean_lines: list[str] = []
    for line in tree_str.split("\n"):
        # remove statictext if the content already appears in the previous line
        if "statictext" in line.lower():
            prev_lines = clean_lines[-3:]
            pattern = r"\[\d+\] StaticText (.+)"

            match = re.search(pattern, line, re.DOTALL)
            if match:
                static_text = match.group(1)[1:-1]  # remove the quotes
                if static_text and all(
                    static_text not in prev_line for prev_line in prev_lines
                ):
                    clean_lines.append(line)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines)


def tree_loaded_successfully(accessibility_tree: AccessibilityTree) -> bool:
    root = next(
        obj for obj in accessibility_tree if obj["role"]["value"] == "RootWebArea"
    )
    # check if has busy attribute
    busy_attr = next((obj for obj in root["properties"] if obj["name"] == "busy"), None)
    return busy_attr is None


async def process(page: Page, client: CDPSession) -> tuple[str, ObsNodesInfo]:
    browser_info = await fetch_browser_info(page, client)

    # wait until content is not busy anymore
    while True:
        accessibility_tree = await fetch_page_accessibility_tree(
            browser_info,
            client,
            CURRENT_VIEWPORT_ONLY,
        )

        # check if the tree is loaded successfully
        if tree_loaded_successfully(accessibility_tree):
            break
        else:
            # wait for a while
            await page.wait_for_timeout(100)

    content, obs_nodes_info = parse_accessibility_tree(accessibility_tree)

    return (clean_accesibility_tree(content), obs_nodes_info)


def get_element_center(
    obs_nodes_info: ObsNodesInfo, element_id: str
) -> tuple[float, float]:
    try:
        node_info = obs_nodes_info[element_id]
    except KeyError:
        raise ValueError(f"Element with id {element_id} not found")
    node_bound = node_info["union_bound"]
    if node_bound is None:
        raise ValueError("Node bound is None")
    x, y, width, height = node_bound
    center_x = x + width / 2
    center_y = y + height / 2
    return (
        center_x,
        center_y,
    )
