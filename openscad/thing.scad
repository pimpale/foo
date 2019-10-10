difference() {
  union() {
    color([0,1,0]) {
      translate([0,0,70]) {
        sphere(r=10);
      }
    }
    translate([0,0,20]) {
      linear_extrude(height=50) {
        circle(r=10);
      }
    }
    linear_extrude(height=25) {
      circle(r=5);
    }
    color([1,0,0]) {
      linear_extrude(height=3) {
        circle(r=60);
      }
    }
    color([0,0,1]) {
      for(i=[0:12]) {
        rotate(i*30) {
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
    color([1,0,1]) {
      for(i=[0:6]) {
        rotate(i*60) {
          translate([2,0,-30]) {
            rotate([1, 0, 0]) {
              linear_extrude(height=30) {
                circle(d=1);
              }
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
