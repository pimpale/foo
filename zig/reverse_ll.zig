const std = @import("std");

const Node = struct {
    val: i32,
    next: ?*Node,
};

pub fn mkNode(allocator: std.mem.Allocator, val: i32, next: ?*Node) !*Node {
    var n = try allocator.create(Node);
    n.val = val;
    n.next = next;
    return n;
}

pub fn printNode(stream: anytype, head: ?*Node) !void {
    while (head != null) {
        try stream.print(head.val);
        head = head.next;
    }
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();

    const stdout = std.io.getStdOut().writer();

    var ll = try mkNode(gpa, 1, null);

    try printNode(stdout, ll);
    try stdout.print("Hello, {s}!\n", .{"world"});
}


