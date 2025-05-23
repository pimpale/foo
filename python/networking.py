#%%
import ipaddress, itertools, textwrap

current = [
    ipaddress.ip_network(s)
    for s in [
        "172.31.128.32/28",
        "172.31.128.16/28",
        "172.31.128.0/28",
        "172.31.96.64/26",
        "172.31.96.32/28",
        "172.31.96.16/28",
        "172.31.80.0/20",
        "172.31.64.0/20",
        "172.31.48.0/20",
        "172.31.32.0/20",
        "172.31.16.0/20",
        "172.31.0.0/20",
    ]
]
vpc_range = ipaddress.ip_network("172.31.0.0/16")
# Let's see all /20 subnets within /16
free_20 = []
for subnet in vpc_range.subnets(new_prefix=20):
    if not any(subnet.overlaps(s) for s in current):
        free_20.append(subnet)
len(free_20), free_20[:10]

# %%
