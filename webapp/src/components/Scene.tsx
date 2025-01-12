import React, { useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useFBO } from "@react-three/drei";
import { io } from "socket.io-client";

const socket = io("http://localhost:5000"); // Flask backend address

const Scene = () => {
  const { gl, scene, camera } = useThree(); // Access WebGL renderer, scene, and camera
  const fbo = useFBO(4096, 2048); // Create an FBO with a custom resolution

  const captureRenderBuffer = () => {
    // Render the scene into the FBO
    gl.setRenderTarget(fbo);
    gl.render(scene, camera);
    gl.setRenderTarget(null);

    // Extract the rendered pixels from the FBO
    const width = fbo.width;
    const height = fbo.height;
    const buffer = new Uint8Array(width * height * 4);
    gl.readRenderTargetPixels(fbo, 0, 0, width, height, buffer);

    // Convert buffer to a base64 image
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(new Uint8ClampedArray(buffer), width, height);
    ctx.putImageData(imageData, 0, 0);

    // Send the base64 image to Flask
    const base64Image = canvas.toDataURL("image/png");
    socket.emit("rendered_image", { data: base64Image });
    console.log("Buffer sent to server.");
  };

  return (
    <>
      <mesh>
        <boxGeometry />
        <meshStandardMaterial color="blue" />
      </mesh>
      <ambientLight />
      <pointLight position={[10, 10, 10]} />
      {/* <button onClick={captureRenderBuffer} style={{ position: "absolute", top: 10, left: 10, zIndex: 10 }}>
        Capture and Send Render
      </button> */}
    </>
  );
};

export default Scene;
