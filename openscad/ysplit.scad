module cylinder(diameter, length) {
  linear_extrude(height=length) {
    circle(d=diameter);
  }
}

difference() {
  union() {
    bend = 150;
    cylinder(6, 10);
    rotate([bend, 0, 0]) {
      cylinder(6, 11);
      translate([0, 0, 10]) {
        rotate([180-bend, 0, 0]) {
          translate([0,0,-1]) {
            cylinder(6, 10);
          }
        }
      }
    }
    rotate([-bend, 0, 0]) {
      cylinder(6, 10);
      translate([0, 0, 10]) {
        rotate([-(180-bend), 0, 0]) {
          cylinder(6, 10);
        }
      }
    }
  }
  union() {
    cylinder(5, 11);
    rotate([150, 0, 0]) {
      cylinder(5, 10);
    }
    rotate([-150, 0, 0]) {
      cylinder(5, 10);
    }
  }
}

