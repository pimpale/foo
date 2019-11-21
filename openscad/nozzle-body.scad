union() {
  difference() {
    scale([0.4,0.4,0.65]) {
      translate([0,0,0.5]) {
        cube(center=true);
      }
    }
    union() {
      cylinder(h=0.65, r1=0.01, r2=0.06, $fn=100);
      cylinder(h=0.65, r=0.03, $fn=100);
    }
  }
}
