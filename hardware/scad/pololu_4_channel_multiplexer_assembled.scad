module pin(){
    cube([2,2,2]);
    translate([0.8,0.8,-3]) cube([0.4,0.4,11.5]);
    //translate([1,1,-3]) cylinder(h=11.5, d=0.8, $fn=100);
}

module pin_block(x,y){
    for (i = [0 : x-1]) {
        for (j = [0 : y-1]){
            translate([i*2, j*2]) pin();
        }
    }
}

module pololu_4_channel_multiplexer_assembled(){
    difference() {
        union(){
            cube([24,24,1]);
            translate([9.5,0.5,1]) pin_block(4,3);
            translate([13.5,17.5,1]) pin_block(5,3);
            translate([0.5,17.5,1]) pin_block(4,3);
        }
        translate([2.5,2.5, -0.1]) cylinder(h=2.2, d=2, $fn=100);
        translate([21.5,2.5, -0.1]) cylinder(h=2.2, d=2, $fn=100);
    }
}

pololu_4_channel_multiplexer_assembled();