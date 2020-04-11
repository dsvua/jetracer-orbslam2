module base_plate(){
    difference(){
        union(){
            translate([-65,-60,-2]) cube([155,120,4]);
            translate([100,50,-2])  cylinder(h=4, r=10, $fn=100);
            translate([100,-50,-2]) cylinder(h=4, r=10, $fn=100);
            translate([-65,-50,-2]) cylinder(h=4, r=10, $fn=100);
            translate([-65,50,-2])  cylinder(h=4, r=10, $fn=100);
            translate([-75,-50,-2]) cube([10,100,4]);
            translate([90,-60,-2])  cube([10,120,4]);
            translate([100, 32,-2]) cube([5,18,4]);
            translate([100,-50,-2]) cube([5,18,4]);
            translate([-32,57,-2])  cylinder(h=4, r=7, $fn=100);
            translate([-69.5,-32,-2]) cylinder(h=4, r=7, $fn=100);

            // jetson nano mount
            translate([-52,-40,0]) cylinder(h=10, r=3, $fn=100);
            translate([-52,18,0])  cylinder(h=10, r=3, $fn=100);
            translate([34,-40,0])  cylinder(h=10, r=3, $fn=100);
            translate([34,18,0])   cylinder(h=10, r=3, $fn=100);

            //PCA mount
            translate([-12,46,0]) cylinder(h=10, r=3, $fn=100);
            translate([-12,27,0]) cylinder(h=10, r=3, $fn=100);
            translate([42,46,0]) cylinder(h=10, r=3, $fn=100);
            translate([42,27,0]) cylinder(h=10, r=3, $fn=100);

            //Mixer mount
            translate([-24,48,0]) cylinder(h=10, r=3, $fn=100);
            translate([-39,48,0]) cylinder(h=10, r=3, $fn=100);
        };
        
        // ESC power switch access gap
        translate([20,50,-2.1]) cube([30,15,6]);
        translate([20,50,-2.1]) rotate([0,0,45]) cube([11,30,6]);
        translate([50,50,-2.1]) rotate([0,0,45]) cube([15,11,6]);
        
        // plate holes to attach to rc car chassic
        translate([70,51,0])   cylinder(h=5, r=1.8, $fn=100, center=true);
        translate([70,51,0])   cylinder(h=2.1, r1=1.8, r2=3.1, $fn=100);

        translate([-32,57,0])  cylinder(h=5, r=1.8, $fn=100, center=true);
        translate([-32,57,0])  cylinder(h=2.1, r1=1.8, r2=3.1, $fn=100);

        translate([91,-31,0])  cylinder(h=5, r=1.8, $fn=100, center=true);
        translate([91,-31,0])  cylinder(h=2.1, r1=1.8, r2=3.1, $fn=100);

        translate([-69.5,-32,0]) cylinder(h=5, r=1.8, $fn=100, center=true);
        translate([-69.5,-32,0]) cylinder(h=2.1, r1=1.8, r2=3.1, $fn=100);

        translate([-90,30,0])  cylinder(h=5, r=1.8, $fn=100, center=true);
        translate([-90,30,0])  cylinder(h=2.1, r1=1.8, r2=3.1, $fn=100);

        translate([-90,-30,0]) cylinder(h=5, r=1.8, $fn=100, center=true);
        translate([-90,-30,0]) cylinder(h=2.1, r1=1.8, r2=3.1, $fn=100);

        // jetson nano mount
        translate([-52,-40,0]) cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([-52,18,0])  cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([34,-40,0])  cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([34,18,0])   cylinder(h=24, r=1.1, $fn=100, center=true);

        // long hole under jetson
        translate([-32,-11,0])    cylinder(h=10, r=20, $fn=100, center=true);
        translate([-32,-31,-2.1]) cube([52,40,6]);
        translate([19,-11,0])     cylinder(h=10, r=20, $fn=100, center=true);
        
        //PCA mount
        translate([-12,46,0]) cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([-12,27,0]) cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([42,46,0]) cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([42,27,0]) cylinder(h=24, r=1.1, $fn=100, center=true);

        // long hole under PCA
        translate([3,33,0])    cylinder(h=10, r=8, $fn=100, center=true);
        translate([3,25,-2.1]) cube([25,16,6]);
        translate([26,33,0])    cylinder(h=10, r=8, $fn=100, center=true);
        
        // WIFI antennas
        translate([102, 50,0])    cylinder(h=24, r=3.5, $fn=100, center=true);
        translate([102,-50,0])    cylinder(h=24, r=3.5, $fn=100, center=true);
        translate([110,-32.65,0]) cylinder(h=24, r=10, $fn=100, center=true);
        translate([110, 32.65,0]) cylinder(h=24, r=10, $fn=100, center=true);

        //Mixer mount
        translate([-24,48,0]) cylinder(h=24, r=1.1, $fn=100, center=true);
        translate([-39,48,0]) cylinder(h=24, r=1.1, $fn=100, center=true);
        
        // long hole under mixer mount
        translate([-43,32,0])    cylinder(h=10, r=8, $fn=100, center=true);
        translate([-27,32,0])    cylinder(h=10, r=8, $fn=100, center=true);
        translate([-43,24,-2.1]) cube([15,16,6]);

        // jetson battery mounts
        translate([85,-17,-2.1]) cube([6,24,6]);
        translate([88,-17,0])    cylinder(h=10, r=3, $fn=100, center=true);
        translate([88,7,0])      cylinder(h=10, r=3, $fn=100, center=true);

        translate([55,-17,-2.1]) cube([6,24,6]);
        translate([58,-17,0])    cylinder(h=10, r=3, $fn=100, center=true);
        translate([58,7,0])      cylinder(h=10, r=3, $fn=100, center=true);

        // holes under battery
        translate([72,28,0])  cylinder(h=10, r=10, $fn=100, center=true);
        translate([72,-42,0]) cylinder(h=10, r=10, $fn=100, center=true);

        // Intel D435i mount
        //front
        //translate([-53,-32.5,-2.1]) cylinder(h=4.2, r1=3.5, r2=1.8, $fn=100);
        //translate([-53, 12.5,-2.1]) cylinder(h=4.2, r1=3.5, r2=1.8, $fn=100);
        //back
        translate([ 91, 25.5,-2.1]) cylinder(h=4.2, r1=3.5, r2=1.8, $fn=100);
        translate([ 52, 25.5,-2.1]) cylinder(h=4.2, r1=3.5, r2=1.8, $fn=100);
        translate([ 52,-31.5,-2.1]) cylinder(h=4.2, r1=3.5, r2=1.8, $fn=100);
        // 92,-32,0
        
        // jeneral holes to save plastic
        //translate([-53,2,0])  #cylinder(h=10, r=10, $fn=100, center=true);
        //translate([-53,-25,0]) #cylinder(h=10, r=10, $fn=100, center=true);
        //translate([-78,-10,0]) #cylinder(h=10, r=10, $fn=100, center=true);
        
        // cut corner
        translate([-75,-21,-2.1]) rotate([0,0,-20])
                translate([-31,0,0]) cube([31,100,6]);
    };
};

base_plate();
//translate([-52,-40,10]) import("../Downloads/jetson nano.stl");
//translate([-80,-18, 2]) cube([20,40,40]);
