import React, { useEffect, useState } from 'react';
import { useRecoilValue } from 'recoil';
import { cameraAngle } from "../state/video";
import * as THREE from "three";

export const CameraPos = (props) => {

    const currCameraAngle = useRecoilValue(cameraAngle);
    // const currSlamKeypoints = useRecoilValue(slamKeypoints);
    const cnvsRef = React.useRef(null);

    const [scene, setScene] = useState(new THREE.Scene());
    // const light = new THREE.DirectionalLight();
    // const camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );
    // light.position.set(0,0,1);
    // scene.add(light);
    const [camera, setCamera] = useState(new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 ));
    const [light, setLight] = useState(new THREE.DirectionalLight());
    const [renderer, setRenderer] = useState();
    const pyramidGeom = new THREE.ConeGeometry(5, 1, 4)
    // const material = new THREE.MeshBasicMaterial( {color: 0xffff00} );
    const [pyramid, setPiramid] = useState(new THREE.Group());
    const [prevAngles, setPrevAngles] = useState({ax: 0, ay:0, az:0});


    useEffect(() => {
        let tmp_renderer = new THREE.WebGLRenderer({
            canvas: cnvsRef.current,
            antialias: true
        });
        camera.position.z = 10;
        tmp_renderer.setSize( window.innerWidth, window.innerHeight );
        setRenderer(tmp_renderer);
        light.position.set(0,0,1);
        scene.add(light);
        const lineMaterial = new THREE.LineBasicMaterial( { color: 0xffffff, transparent: true, opacity: 0.5 } );
        const meshMaterial = new THREE.MeshPhongMaterial( { color: 0x156289, emissive: 0x072534, side: THREE.DoubleSide, flatShading: true } );
        pyramid.add(new THREE.LineSegments( pyramidGeom, lineMaterial ));
        pyramid.add(new THREE.Mesh( pyramidGeom, meshMaterial ));
        scene.add(pyramid);

        console.log("pyramidGeom", pyramidGeom);
        console.log("pyramid", pyramid);

        let animate = function () {
            requestAnimationFrame( animate );
            tmp_renderer.render( scene, camera );
        };
        animate();      

    }, []);
    
    useEffect(() => {
        // console.log("currCameraAngle", currCameraAngle, "pyramid", pyramid);
        if (currCameraAngle && pyramid)
        {
            // const eu = new THREE.Euler( currCameraAngle.ax * 180 / Math.PI,
            //     currCameraAngle.ay * 180 / Math.PI,
            //     currCameraAngle.az * 180 / Math.PI,
            //     'XYZ' );
            const eu = new THREE.Euler( currCameraAngle.ax * Math.PI / 180,
                currCameraAngle.ay * Math.PI / 180,
                (currCameraAngle.az + 90) * Math.PI / 180,
                'XYZ' );
            pyramid.setRotationFromEuler(eu);
            console.log("Euler", eu, "currCameraAngle", currCameraAngle);
            // setPrevAngles(currCameraAngle);
        }
    }, [currCameraAngle]);

    return (
        <div>
            <canvas ref={cnvsRef} width="848" height="480" id="cnvs" />
        </div>
    )
}