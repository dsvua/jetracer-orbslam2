import React, { useState, useEffect, useRef } from 'react';
import { useRecoilValue, useRecoilState } from 'recoil';
import { videoImage, videoImageParams } from "../state/video";
import { connectionStateIsConnected } from '../state/network'
// import { slamKeypoints } from '../state/slam'


export const VideoFeed = (props) => {

    const currVideoImage = useRecoilValue(videoImage);
    const currVideoImageParams = useRecoilValue(videoImageParams);
    // const currSlamKeypoints = useRecoilValue(slamKeypoints);
    const isConnected = useRecoilValue(connectionStateIsConnected);

    const canvasRef = useRef(null);

    // useEffect(() => {
    //     const canvas = canvasRef.current;
    //     canvas.width = currVideoImageParams.width;
    //     canvas.height = currVideoImageParams.height;
    //     const context = canvas.getContext('2d');
    //     //Our first draw
    //     context.fillStyle = '#000000';
    //     context.fillRect(0, 0, context.canvas.width, context.canvas.height);
    // }, []);
    
    useEffect(() => {

        if (isConnected){

            let blob = new Blob([currVideoImage.image], {type: "image/jpeg"});
            let urlCreator = window.URL || window.webkitURL;
            let imageUrl = urlCreator.createObjectURL( blob );
            let img = document.querySelector( "#videoframe" );
            img.src = imageUrl;
        
        }
    }, [currVideoImage]);

    return (
        <div>
            <img id="videoframe" {...props} />
        </div>
    )
}