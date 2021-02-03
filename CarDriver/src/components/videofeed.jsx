import React, { useState, useEffect, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import { videoImage, videoImageParams } from "../state/video";
import { connectionStateIsConnected } from '../state/network'


export const VideoFeed = (props) => {

    const currVideoImage = useRecoilValue(videoImage);
    const currVideoImageParams = useRecoilValue(videoImageParams);
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
    
    useEffect(() => {

        if (isConnected){
            const imageSize = currVideoImageParams.height * currVideoImageParams.width;
            const clampedImage = new Uint8ClampedArray(imageSize * 4);
            for (let i=0; i < imageSize; i++) {
                clampedImage[i*4] = currVideoImage[i];
                clampedImage[i*4+1] = currVideoImage[i];
                clampedImage[i*4+2] = currVideoImage[i];
                clampedImage[i*4+3] = 255;
            };
            const imgData = new ImageData(clampedImage, currVideoImageParams.width, currVideoImageParams.height);
    
            const canvas = canvasRef.current;
            canvas.width = currVideoImageParams.width;
            canvas.height = currVideoImageParams.height;
            const context = canvas.getContext('2d');
            context.putImageData(imgData, 0, 0); 
            // context.putImageData(imgData, 0, 0, 0, 0, currVideoImageParams.width, currVideoImageParams.height); 
            // console.log("Draw new image, pixel:", currVideoImage[848*200 + 200]);   
        }
    }, [currVideoImage]);

    return (
        <div>
            {/* {isConnected ? <canvas ref={canvasRef} {...props} style={{width: "100%"}} /> : ""} */}
            <canvas ref={canvasRef} {...props} />
        </div>
    )
}