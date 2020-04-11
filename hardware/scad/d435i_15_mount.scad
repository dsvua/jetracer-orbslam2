difference(){
    union (){
        translate([0,0,0]) cube ([13,3,14]);
        rotate([0,0,15]) {
            translate([0,0,0]) cube ([3,60,14]);
            translate([1.5,60,0]) cylinder (h=14, r=1.5, $fn=100);
        }
        //translate([3,3,0]) cylinder (h=14, r=3, $fn=100);
        translate([13,1.5,0]) cylinder (h=14, r=1.5, $fn=100);
    }
    //translate([6,6,-1]) cylinder (h=16, r=3, $fn=100);
    //translate([5,2,-1]) cube ([30,6,16]);
    //translate([2,5,-1]) cube ([6,30,16]);
    translate([7,5,7]) rotate([90,0,0])   cylinder (h=16, r=1.9, $fn=100);
        rotate([0,0,15]) {
            translate([5,16,7]) rotate([0,-90,0]) cylinder (h=16, r=1.9, $fn=100);
            translate([5,26,7]) rotate([0,-90,0]) cylinder (h=16, r=1.9, $fn=100);
            translate([5,36,7]) rotate([0,-90,0]) cylinder (h=16, r=1.9, $fn=100);
            translate([5,46,7]) rotate([0,-90,0]) cylinder (h=16, r=1.9, $fn=100);
            translate([5,56,7]) rotate([0,-90,0]) cylinder (h=16, r=1.9, $fn=100);
        }
};


