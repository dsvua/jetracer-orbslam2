import './App.css';
import { WsConnection } from './components/connection';
import React, {useState, useEffect} from "react";
import useWebsocket from "./hooks/useWebsockets";
import { Feed } from "./components/feed";

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
    <div style={{position: 'relative'}}
         className="container-fluid d-flex flex-column mx-0 px-0 min-vh-100 h-100 w-100">
          <Feed />        
      <div style={{position: 'absolute', top:0, left: 0, zIndex:2}}
           className="container-fluid d-flex flex-column min-vh-100 h-100 w-100">
        <WsConnection />        
      </div>
    </div>
  );
}

export default App;
