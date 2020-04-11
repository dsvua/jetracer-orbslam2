module battery_mount() {
    x=-5;
    y=-1;
    z=14;
    difference(){
        union(){
            translate([-7,0,0]) cube([12,18,4]);
            cylinder(h=4, r=7, $fn=100);
            translate([97,-6,0]) cube([9,30,4]);
            translate([102,-6,0]) cylinder(h=4, r=5.5, $fn=100);
            translate([x-2, y+23,z-30]) rotate ([0,90,0])
                    cylinder(h=12, d=37, $fn=100);
            translate([x+30, y+23,z-30]) rotate ([0,90,0])
                    cylinder(h=12, d=37, $fn=100);
            translate([x+65, y+23,z-30]) rotate ([0,90,0])
                    cylinder(h=12, d=37, $fn=100);
            translate([x+99, y+23,z-30]) rotate ([0,90,0])
                    cylinder(h=12, d=37, $fn=100);
            translate([x+105, y+23,z-30]) rotate([-45,0,0]) translate([0, 0,17])
                    cube([12,16,6], center = true);
            translate([x-2, y+19,0]) cube([113,8,12]);
            translate([x-2, y+19,z-56]) cube([113,8,12]);
            
            translate([x+30, y+9.7,z-53]) cube([12,15,15]);
            translate([x+65, y+5,z-53]) cube([12,20,10]);
            translate([x+65, y+11,z-49.85]) rotate([45,0,0]) cube([12,5,8.53]);
        }
        translate([x+64, y,z-50]) cube([14,5,8.53]);
        translate([x+29, y+11,z-58]) cube([14,5,8]);
        translate([x+64, y+6,z-58]) cube([14,5,8]);
        translate([0,0,-1]) cylinder(h=6, r=1.8, $fn=100);
        translate([102,-6,-1]) cylinder(h=6, r=1.8, $fn=100);
        union() {
            translate([x+60, y+23,z-30]) cube([150,2,70], center = true);
            translate([x+5, y+35,9]) rotate([90,0,0]) 
                    cylinder(h=20, d=3.3, $fn=100);
            translate([x+104, y+35,9]) rotate([90,0,0]) 
                    cylinder(h=20, d=3.3, $fn=100);
            translate([x+5, y+35,z-52]) rotate([90,0,0]) 
                    cylinder(h=20, d=3.3, $fn=100);
            translate([x+104, y+35,z-52]) rotate([90,0,0]) 
                    cylinder(h=20, d=3.3, $fn=100);
            translate([x+15, y+27,z-46]) cube([80,34,35]);
            translate([x+15, y+27,z-50]) cube([80,14,34]);
            translate([x-10, y+23,z-30]) rotate([0,90,0]) 
                    cylinder(h=130, d=20, $fn=100);
            
            // model of the battery
            translate([x, y+23,z-30]) rotate([0,90,0]) 
                    cylinder(h=110, d=33, $fn=100);
            translate([x+91.5, y+23,z-30]) rotate([-45,0,0]) translate([0, 0,16])
                    cube([35,12,6], center = true);
        }
    }
}

module battery_mount1() {
    x=-5;
    y=-1;
    z=14;
    difference() {
        battery_mount();
        translate([x-10, y+23,z-57]) cube([130,30,60]);
    }
}
module battery_mount2() {
    x=-5;
    y=-1;
    z=14;
    difference() {
        battery_mount();
        translate([x-10, y-12,z-57]) cube([130,35,60]);
    }
}

battery_mount();
//rotate([-90,0,0]) battery_mount1();
//translate([0,70,0]) rotate([90,0,0]) battery_mount2();
