import './App.css';
import { WsConnection } from './components/connection';
import React, {useState, useEffect} from "react";
import useWebsocket from "./hooks/useWebsockets";
import { VideoFeed } from "./components/videofeed";

function getWindowDimensions() {
  const { innerWidth: width, innerHeight: height } = window;
  return {
      width,
      height
  };
}

const App = () => {
  const [windowDimensions, setWindowDimensions] = useState(getWindowDimensions());

  useWebsocket();

  function handleResize() {
    // console.log("before resize", getWindowDimensions());
    setWindowDimensions(getWindowDimensions());
    // console.log("after resize", getWindowDimensions());
  }

  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
      <div className="container-fluid d-flex flex-column min-vh-100 h-100 w-100">
        <div className="row align-items-start w-100">
          {/* <div className="col"> */}
            <VideoFeed />        
          {/* </div> */}
        </div>
        {/* <div className="row align-items-end w-100">
          <div className="col"> */}
            <WsConnection />        
          {/* </div>
        </div> */}
      </div>
  );
}

export default App;
