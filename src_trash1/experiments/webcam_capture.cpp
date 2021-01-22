#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <fstream>
#include <string>
#include <poll.h>

using namespace std;

int main() {

    cout << "Open the device" << endl;

    // 1.  Open the device
    int fd; // A file descriptor to the video device
    fd = open("/dev/video0",O_RDWR);
    if(fd < 0){
        perror("Failed to open device, OPEN");
        return 1;
    }

    struct pollfd fds[1];
    fds[0].fd = fd;
    fds[0].events = POLLIN;

    cout << "Ask the device if it can capture frames" << endl;

    // 2. Ask the device if it can capture frames
    v4l2_capability capability;
    if(ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0){
        // something went wrong... exit
        perror("Failed to get device capabilities, VIDIOC_QUERYCAP");
        return 1;
    }

    cout << "Set Image format" << endl;

    // 3. Set Image format
    v4l2_format imageFormat;
    imageFormat.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    imageFormat.fmt.pix.width = 1280;
    imageFormat.fmt.pix.height = 720;
    imageFormat.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10;
    imageFormat.fmt.pix.field = V4L2_FIELD_NONE;
    // tell the device you are using this format
    if(ioctl(fd, VIDIOC_S_FMT, &imageFormat) < 0){
        perror("Device could not set format, VIDIOC_S_FMT");
        return 1;
    }

    cout << "Request Buffers from the device" << endl;

    // 4. Request Buffers from the device
    v4l2_requestbuffers requestBuffer = {0};
    requestBuffer.count = 1; // one request buffer
    requestBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // request a buffer wich we an use for capturing frames
    requestBuffer.memory = V4L2_MEMORY_MMAP;

    if(ioctl(fd, VIDIOC_REQBUFS, &requestBuffer) < 0){
        perror("Could not request buffer from device, VIDIOC_REQBUFS");
        return 1;
    }

    cout << "Query the buffer to get raw data" << endl;

    // 5. Query the buffer to get raw data ie. ask for the you requested buffer
    // and allocate memory for it
    v4l2_buffer queryBuffer = {0};
    queryBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    queryBuffer.memory = V4L2_MEMORY_MMAP;
    queryBuffer.index = 0;
    if(ioctl(fd, VIDIOC_QUERYBUF, &queryBuffer) < 0){
        perror("Device did not return the buffer information, VIDIOC_QUERYBUF");
        return 1;
    }
    // use a pointer to point to the newly created buffer
    // mmap() will map the memory address of the device to
    // an address in memory
    char* buffer = (char*)mmap(NULL, queryBuffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                        fd, queryBuffer.m.offset);
    memset(buffer, 0, queryBuffer.length);

    cout << "Get a frame" << endl;

    // 6. Get a frame
    // Create a new buffer type so the device knows whichbuffer we are talking about
    v4l2_buffer bufferinfo;
    memset(&bufferinfo, 0, sizeof(bufferinfo));
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    // bufferinfo.index = 0;

    cout << "Step Activate streaming" << endl;

    // Activate streaming
    int type = bufferinfo.type;
    if(ioctl(fd, VIDIOC_STREAMON, &type) < 0){
        perror("Could not start streaming, VIDIOC_STREAMON");
        return 1;
    }

/***************************** Begin looping here *********************/
    cout << "Queue the buffer" << endl;
    // Queue the buffer
    if(ioctl(fd, VIDIOC_QBUF, &bufferinfo) < 0){
        perror("Could not queue buffer, VIDIOC_QBUF");
        return 1;
    }

    cout << "Polling ...." << endl;
    while (poll(fds, 1, 5000) > 0 && !quit) {
        cout << "Dequeue the buffer" << endl;
        if (fds[0].revents & POLLIN) {
            // Dequeue the buffer
            if(ioctl(fd, VIDIOC_DQBUF, &bufferinfo) < 0){
                perror("Could not dequeue the buffer, VIDIOC_DQBUF");
                return 1;
            }
            // Frames get written after dequeuing the buffer

            cout << "Buffer has: " << (double)bufferinfo.bytesused / 1024
                    << " KBytes of data" << endl;

            cout << "Write the data out to file" << endl;
            // Write the data out to file
            // ofstream outFile;
            // outFile.open("/mnt/ssd/webcam_output.jpeg", ios::binary| ios::app);
            // Write the data out to file
            // outFile.write(buffer, (double)bufferinfo.bytesused);
            // Close the file
            // outFile.close();
            if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &bufferinfo))
                ERROR_RETURN("Failed to queue camera buffers: %s (%d)",
                    strerror(errno), errno);

        }
    }


/******************************** end looping here **********************/

    // end streaming
    if(ioctl(fd, VIDIOC_STREAMOFF, &type) < 0){
        perror("Could not end streaming, VIDIOC_STREAMOFF");
        return 1;
    }

    close(fd);
    return 0;
}
