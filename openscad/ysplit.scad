union() {

  rotate([0, 90, 0]) {
    translate([0, 0, 2]) {
      cylinder(h=10);
    }
    translate([0, -2, -10]) {
      rotate([0, 0, 54]) {
        cylinder(h=10);
      }
    }
    translate([0, 2, -10]) {
      rotate([0, 0, -54]) {
        cylinder(h=10);
      }
    }
  }
  rotate_extrude(angle = 90) {
    translate([2, 0, 0]) {
      circle(r=1);
    }
  }

  rotate_extrude(angle = -90) {
    translate([2, 0, 0]) {
      circle(r=1);
    }
  }
}
