module hollow_cylinder(outer, inner, length) {
  linear_extrude(height=length) {
    difference() {
      circle(d=outer);
      circle(d=inner);
    }
  }
}

module cylinder(diameter, length) {
  linear_extrude(height=length) {
    circle(d=diameter);
  }
}

difference() {
  union() {
    cylinder(6, 10);
    rotate([150, 0, 0]) {
      cylinder(6, 10);
    }
    rotate([-150, 0, 0]) {
      cylinder(6, 10);
    }
  }
  union() {
    cylinder(5, 11);
    rotate([150, 0, 0]) {
      cylinder(5, 11);
    }
    rotate([-150, 0, 0]) {
      cylinder(5, 11);
    }
  }
}

