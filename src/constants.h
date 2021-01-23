
#ifndef JETRACER_CONSTANTS_H
#define JETRACER_CONSTANTS_H

namespace Jetracer {

    #define CAPACITY 5 // allow max latency of 5 frames in a queue
    #define GPU_CAPACITY 3 // schedule 3 frames in parallel


    // The nomenclature is for the ESC throttle, for the steering neutral is center,
    // full forward is right, full reverse is left

    // default PWM for general ESC/Servo
    // #define PWM_FULL_REVERSE 204 // 1ms/20ms * 4096
    // #define PWM_NEUTRAL 307      // 1.5ms/20ms * 4096
    // #define PWM_FULL_FORWARD 409 // 2ms/20ms * 4096

    // PWM between 304-327 are neutral with my ESC
    // values below tuned for my ESC and remote transmitter/receiver
    #define ESC_PWM_FULL_REVERSE 100 // 1ms/20ms * 4096
    #define ESC_PWM_NEUTRAL 307      // 1.5ms/20ms * 4096
    #define ESC_PWM_MINIMAL_FORWARD 328 // motor does not turn with smaller values
    #define ESC_PWM_FULL_FORWARD 420 // 2ms/20ms * 4096
    #define ESC_REVERSE_RANGE (ESC_PWM_NEUTRAL - ESC_PWM_FULL_REVERSE)
    #define ESC_FORWARD_RANGE (ESC_PWM_FULL_FORWARD - ESC_PWM_NEUTRAL)

    // tuned for my particular servo
    #define SERVO_PWM_FULL_LEFT 273 // 1ms/20ms * 4096
    #define SERVO_PWM_NEUTRAL 313      // 1.5ms/20ms * 4096
    #define SERVO_PWM_FULL_RIGHT 353 // 2ms/20ms * 4096
    #define SERVO_LEFT_RANGE (SERVO_PWM_NEUTRAL - SERVO_PWM_FULL_LEFT)
    #define SERVO_RIGHT_RANGE (SERVO_PWM_FULL_RIGHT - SERVO_PWM_NEUTRAL)


    // The Steering Servo is plugged into the follow PWM channel
    #define STEERING_CHANNEL 0
    // The ESC is plugged into the following PWM channel
    #define ESC_CHANNEL 1
}

#endif // JETRACER_CONSTANTS_H