import { useRef, useImperativeHandle, forwardRef, useCallback } from "react";
import { Environment, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";

const Main = forwardRef((props, ref) => {
  const box = useRef();
  const FBOcamera = useRef();

  useImperativeHandle(ref, () => ({
    camera: FBOcamera.current,
  }));

  useFrame(({ clock }) => {
    box.current.rotation.x += 0.01;
    box.current.rotation.z += 0.01;
  });

  const handlePointer = useCallback((hover) => {
    document.body.style.cursor = hover ? "pointer" : "default";
  }, []);

  return (
    <group ref={ref}>
      <OrbitControls camera={FBOcamera.current} />
      <PerspectiveCamera near={0.001} position={[0, 0, 10]} fov={35} ref={FBOcamera} makeDefault={true} />
      <ambientLight intensity={1} />

      <mesh
        ref={box}
        onPointerEnter={() => handlePointer(true)}
        onPointerOut={() => handlePointer(false)}
        position={[-1, 0.5, 0]}
      >
        <icosahedronGeometry args={[0.8, 2]} />
        <meshStandardMaterial color={"rgb(0, 256, 0)"} roughness={0.1} flatShading={true} />
      </mesh>
      <mesh
        ref={box}
        onPointerEnter={() => handlePointer(true)}
        onPointerOut={() => handlePointer(false)}
        position={[1, 1, 0]}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color={"rgb(256, 0, 0)"} roughness={0.1} flatShading={true} />
      </mesh>
      <mesh
        position={[0, 0, 0]} // Adjust position to place it below the box
        rotation={[-Math.PI / 2, 0, 0]} // Rotate to make it flat
      >
        <planeGeometry args={[100, 100]} /> // Large plane
        <meshStandardMaterial color={"brown"} side={2} />
      </mesh>
    </group>
  );
});

export default Main;
