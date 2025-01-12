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
      <PerspectiveCamera near={0.001} position={[0, 0, 2]} fov={100} ref={FBOcamera} makeDefault={true} />
      <mesh ref={box} onPointerEnter={() => handlePointer(true)} onPointerOut={() => handlePointer(false)}>
        <icosahedronGeometry args={[0.8, 2]} />
        <meshStandardMaterial color={0xcccccc} roughness={0.1} flatShading={true} />
      </mesh>
      <mesh ref={box} onPointerEnter={() => handlePointer(true)} onPointerOut={() => handlePointer(false)}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color={0xcccccc} roughness={0.1} flatShading={false} />
      </mesh>
    </group>
  );
});

export default Main;
