import React, { useState, useEffect, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import { videoImage, videoImageParams } from "../state/video";
import { connectionStateIsConnected } from '../state/network'
// import { slamKeypoints } from '../state/slam'


export const VideoFeed = (props) => {

    const currVideoImage = useRecoilValue(videoImage);
    const currVideoImageParams = useRecoilValue(videoImageParams);
    // const currSlamKeypoints = useRecoilValue(slamKeypoints);
    const isConnected = useRecoilValue(connectionStateIsConnected);

    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        canvas.width = currVideoImageParams.width;
        canvas.height = currVideoImageParams.height;
        const context = canvas.getContext('2d');
        //Our first draw
        context.fillStyle = '#000000';
        context.fillRect(0, 0, context.canvas.width, context.canvas.height);
    }, []);
    
    const paint_circle = (ctx, x, y) => {
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 360);
        ctx.fillStyle = "green";
        ctx.fill();
    }

    useEffect(() => {

        if (isConnected){
            const imageSize = currVideoImageParams.height * currVideoImageParams.width;
            const canvas = canvasRef.current;
            canvas.width = currVideoImageParams.width;
            canvas.height = currVideoImageParams.height;
            const context = canvas.getContext('2d');

            const clampedImage = new Uint8ClampedArray(imageSize * 4);


            for (let i=0; i < imageSize; i++) {
                clampedImage[i*4] = currVideoImage.image[i];
                clampedImage[i*4+1] = currVideoImage.image[i];
                clampedImage[i*4+2] = currVideoImage.image[i];
                clampedImage[i*4+3] = 255;
            };

            // paint keypoints
            // let floats_x = new Float32Array(currVideoImage.x.buffer);
            // let floats_y = new Float32Array(currVideoImage.y.buffer);
            let floats_x = new Uint16Array(currVideoImage.x.buffer);
            let floats_y = new Uint16Array(currVideoImage.y.buffer);
            let index = 0;

            for (let i=0; i<floats_x.length; i++){
                // paint_circle(context, floats_x[i], floats_y[i]);
                // drawPixel(canvasData, canvas.width, floats_x[i], floats_y[i], 255);
                index = (floats_x[i] + floats_y[i] * canvas.width) * 4;
                clampedImage[index + 1] = 255;
                clampedImage[index + 5] = 255;
                clampedImage[index - 5] = 255;
                index = (floats_x[i] + (floats_y[i] - 1) * canvas.width) * 4;
                clampedImage[index + 1] = 255;
                index = (floats_x[i] + (floats_y[i] + 1) * canvas.width) * 4;
                clampedImage[index + 1] = 255;


                // if (floats_y > 350) {
                //     console.log("i", i, "x", floats_x, "y", floats_y);
                // }
                // if (i === 100){
                //     console.log("i, x, y, length", i, floats_x[i], floats_y[i], floats_x.length, 
                //     currVideoImage.x[i*4], currVideoImage.x[i*4+1], currVideoImage.x[i*4+2], currVideoImage.x[i*4+3]);
                // }
            }

            const imgData = new ImageData(clampedImage, currVideoImageParams.width, currVideoImageParams.height);

            context.putImageData(imgData, 0, 0);

        }
    }, [currVideoImage]);

    return (
        <div>
            {/* {isConnected ? <canvas ref={canvasRef} {...props} style={{width: "100%"}} /> : ""} */}
            <canvas ref={canvasRef} {...props} />
        </div>
    )
}