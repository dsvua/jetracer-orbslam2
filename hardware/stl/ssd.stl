
// sizes in mm: 75, 57.3, 10.5

module ssd_t5(){
    difference(){
        dy = 57.3 - 10.5;
        hz = 10.5/2;
        union(){
            cube([75, dy, 10.5]);
            translate([0, 0, hz]) rotate([0,90,0]) cylinder(h=75, d=10.5);
            translate([0, dy, hz]) rotate([0,90,0]) 
                    cylinder(h=75, d=10.5);
        }
        translate([-1, dy/2, hz]) cube([10,10,3], center=true);
    }
}

ssd_t5();