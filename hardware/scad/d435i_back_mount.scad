difference(){
    union(){
        cylinder(h=5, d=44, $fn=100);
        translate([-22,0,0]) cube([44,15,5]);
        translate([0,-58,0]) cube([3,40,5]);
        translate([-22,10,0]) cube([5,5,13]);
        translate([ 17,10,0]) cube([5,5,13]);
        translate([1.5,-58,0]) cylinder(h=5, d=3, $fn=100);
        
    }
    translate([0,0,-1]) cylinder(h=7, d=34, $fn=100);
    translate([-17,0,-1]) cube([34,30,7]);
    rotate([0,90,0]) {
        translate([-2.5,-30,-4]) cylinder(h=8, r=1.8, $fn=100);
        translate([-2.5,-36,-4]) cylinder(h=8, r=1.8, $fn=100);
        translate([-2.5,-42,-4]) cylinder(h=8, r=1.8, $fn=100);
        translate([-2.5,-48,-4]) cylinder(h=8, r=1.8, $fn=100);
        translate([-2.5,-54,-4]) cylinder(h=8, r=1.8, $fn=100);
    }
    rotate([90,0,0]) {
        translate([-19.5,8.5,-16]) cylinder(h=8, r=1.8, $fn=100);
        translate([ 19.5,8.5,-16]) cylinder(h=8, r=1.8, $fn=100);
    }
}