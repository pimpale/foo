difference() {
    hull() {
        translate([5,5,0]) {
            cylinder(35, r=5);
        }
        translate([130,5,0]) {
            cylinder(35, r=5);
        }
        translate([130,60,0]) {
            cylinder(35, r=5);
        }
        translate([5,60,0]) {
            cylinder(35, r=5);
        }
    }
    translate([5, 5, 5]){
        cube([125, 55, 35]);
    }
    translate([-1, 25, 15]) {
       cube([15, 15, 7.5]);
    }
}
 

peg(10+2.5, 10+2.5, 2);
peg(10+2.5, 50+2.5, 2);
peg(120+2.5, 10+2.5, 2);
peg(120+2.5, 50+2.5, 2);

module peg(x, y, z,) {
    translate([x, y, z]) {
        difference() {
            cylinder(18, r=5);
            translate([0,0,5]){
                cylinder(15, r=2.5);
            }
        }
    }
}