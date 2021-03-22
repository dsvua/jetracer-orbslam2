import React, { useEffect, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import { videoImage, videoImageParams } from "../state/video";
import { connectionStateIsConnected } from '../state/network'
import * as THREE from "three";

export const VideoFeed = (props) => {

    const currVideoImage = useRecoilValue(videoImage);
    const currVideoImageParams = useRecoilValue(videoImageParams);
    // const currSlamKeypoints = useRecoilValue(slamKeypoints);
    const isConnected = useRecoilValue(connectionStateIsConnected);
   
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
            <img style={{resizeMode: 'contain'}} 
            id="videoframe" 
            className="container-fluid d-flex flex-column mx-0 px-0 min-vh-100 h-100 w-100"
            {...props} />
        </div>
    )
}