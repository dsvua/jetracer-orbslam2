#include <stdio.h>
#include <stdlib.h>
#include "gstCamera.h"
#include <signal.h>

bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT )
    {
        printf("received SIGINT\n");
        signal_recieved = true;
    }
}

int main( int argc, char** argv )
{
    gstCamera* camera = gstCamera::Create(cmdLine.GetInt("width", gstCamera::DefaultWidth),
                                   cmdLine.GetInt("height", gstCamera::DefaultHeight),
                                   cmdLine.GetString("camera"));

    if( !camera ){
            printf("\nCamera:  failed to initialize camera device\n");
            return 0;
    }

    cout << "Camera detected" << endl;

    if( !camera->Open() )
    {
        printf("Camera:  failed to open camera for streaming\n");
        return 0;
    }
    
    printf("Camera:  camera open for streaming\n");
    float* imgRGBA = NULL;

    if( !camera->CaptureRGBA(&imgRGBA, 1000) )
            printf("detectnet-camera:  failed to capture RGBA image from camera\n");

    cout << "Picture is captured" << endl;


}