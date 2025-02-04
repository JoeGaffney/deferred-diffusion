import React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import Scene from "../components/Scene";
import FBO from "../components/FBO";

const Home: React.FC = () => {
  return (
    <div style={{ width: "90%", height: "90%", border: "2px solid gray", backgroundColor: "black" }}>
      <Canvas shadows>
        {/* <OrbitControls />
        <ambientLight />
        <pointLight position={[10, 10, 10]} />
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="orange" />
        </mesh>
        <Scene /> */}
        {/* <FBO /> */}
        <FBO />
      </Canvas>
    </div>
  );
};

export default Home;
