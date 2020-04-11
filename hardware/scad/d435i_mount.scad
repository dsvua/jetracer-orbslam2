// sizes in mm: 75, 57.3, 10.5


module d435i_mount(){
    difference(){
        union (){
            translate([3,0,0]) cube([13,40.8,2]);
            translate([3,0,12]) cube([13,40.8,2]);
            translate([3,0,0]) cube ([13,3,14]);
            translate([0,3,0]) cube ([3,60,14]);
            translate([3,3,0]) cylinder (h=14, r=3, $fn=100);
            translate([16,1.5,0]) cylinder (h=14, r=1.5, $fn=100);
            translate([1.5,63,0]) cylinder (h=14, r=1.5, $fn=100);

            translate([-13,51,0]) cube([13,8.5,14]);
            translate([-11.5,59.5,0]) cube([11.5,1.5,14]);
            translate([-3,61,0]) cube([3,3,14]);
            translate([-11.5,59.5,0]) cylinder (h=14, r=1.5, $fn=100);
            translate([-11.5,59.5,0]) cylinder (h=14, r=1.5, $fn=100);
            translate([-12.5,51,0]) cylinder (h=14, r=0.5, $fn=100);
        }
        translate([16,3,-1]) rotate([0,0,19]) cube([13,40,4]);
        translate([16,3,11]) rotate([0,0,19]) cube([13,40,4]);
        translate([6,6,2]) cylinder (h=10, r=3, $fn=100);
        translate([9,5,7]) rotate([90,0,0]) cylinder (h=16, r=1.7, $fn=100);
        //translate([5,18,7]) rotate([0,-90,0]) cylinder (h=16, r=1.7, $fn=100);
        //translate([5,28,7]) rotate([0,-90,0]) cylinder (h=16, r=1.7, $fn=100);
        //translate([5,38,7]) rotate([0,-90,0]) cylinder (h=16, r=1.7, $fn=100);
        translate([3.1,48,7]) rotate([0,-90,0]) 
                cylinder (h=3.2, r1=1.6, r2=3.3, $fn=100);
        //translate([5,58,7]) rotate([0,-90,0]) cylinder (h=16, r=1.7, $fn=100);

        translate([-6,52.3,-0.1]) cylinder (h=14.2, r=6, $fn=100);
        translate([-2.9,64,-0.1]) cylinder (h=14.2, r=3, $fn=100);
        translate([-1,42.3,-0.1]) cube ([1,10,14.2]);
        translate([-12,42.3,-0.1]) cube ([1,10,14.2]);
    };
}

d435i_mount();

