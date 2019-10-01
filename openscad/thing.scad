difference() {
  union() {
    linear_extrude(height=25) {
      circle(r=5);
    }
    linear_extrude(height=3) {
      circle(r=50);
    }
    for(i=[0:6]) {
      rotate(i*60) {
        translate([3,0,-30]) {
          rotate([10, 0, 0]) {
            linear_extrude(height=30) {
              circle(d=1);
            }
          }
        }
      }
    }
  }
  for(i=[0:6]) {
    rotate(i*60) {
      translate([2,0,0]) {
        linear_extrude(height=35) {
          circle(d=1);
        }
      }
    }
  }
}
