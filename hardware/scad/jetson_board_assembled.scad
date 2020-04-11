
use <battery_mount.scad>;
use <jetracer_board_no_battery.scad>;
use <d435i_mount.scad>;
use <ssd.scad>;
use <pololu_4_channel_multiplexer_assembled.scad>;

base_plate();
translate([-32,57,2]) color("yellow") battery_mount();
//translate([-32,57,2]) color("yellow") battery_mount2();

translate([-12,-40,10]) color("gray") import("../stl/jetson nano.stl");

dx=-4;
dy=0;
translate([dx-24,dy+5,2]) color("yellow") rotate([90,0,180]) d435i_mount();
translate([dx-24,dy-40,2]) color("yellow") rotate([90,0,180]) d435i_mount();
translate([dx-52,dy-10.5,50]) rotate([90,0,-90]) color("silver") import("../stl/D435i.stl");

translate([34.3,36,11]) rotate([90,0,0]) color("blue") import("../stl/PCA9685.stl");
translate([-1.3,51,10]) rotate([0,0,180]) color("blue")
    pololu_4_channel_multiplexer_assembled();

//SSD
translate([-27,-55.5,2+(10.5/2)]) rotate([90,0,90]) color("aqua") ssd_t5();

// metal spacers rc, car base
translate([70,51,-34]) color("red") cylinder(h=32, d=5.6, $fn=100);
translate([91,-31,-34]) color("red") cylinder(h=32, d=5.6, $fn=100);
translate([-69.5,-32,-34]) color("red") cylinder(h=32, d=5.6, $fn=100);
translate([-32,57,-34]) color("red") cylinder(h=32, d=5.6, $fn=100);
translate([-32,63.1,-49]) color("black") cube([8,3.6,15]);
translate([-24,67.5,-49]) color("black") cube([44,3.6,15]);
translate([ 20,63.1,-49]) color("black") cube([40,3.6,15]);
