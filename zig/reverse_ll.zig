const std = @import("std");
const Allocator = std.mem.Allocator;

const Node = struct {
    val: i32,
    next: ?*Node,
};

pub fn mkNode(allocator:Allocator, val:i32, next: ?*Node) !*Node {
  try allocator.create(Node);
}

pub fn printNode(stream: std.writer ,head:?*Node) !void {
  while(head != null) {
      print(head.val);
      head = head.next;
  }
}


pub fn main() !void {
    const stdout= std.io.getStdOut().writer();

    var ll = try mkNode(Allocator


    printNode(stdout, ll);
    try stdout.print("Hello, {s}!\n", .{"world"});
}
