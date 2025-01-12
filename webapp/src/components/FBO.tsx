import { useMemo, useRef } from "react";
import { useThree, createPortal, useFrame, extend } from "@react-three/fiber";
import { useFBO } from "@react-three/drei";
import { Scene, Color } from "three";
import { button, useControls } from "leva";
import Main from "./Main";

const FBO = () => {
  const { viewport } = useThree();
  const material = useRef();
  const main = useRef();

  const { postprocessing, bgColor } = useControls({
    postprocessing: true,
    bgColor: "#000000",
    testColor: "#000000",
    number: 3,
    sendTarget: button((get) => alert(`Number value is ${get("number").toFixed(2)}`)),
  });

  const FBOscene = useMemo(() => {
    const FBOscene = new Scene();
    return FBOscene;
  }, []);
  const target = useFBO();

  useFrame((state) => {
    if (main.current.camera) {
      state.gl.setRenderTarget(target);
      state.gl.render(FBOscene, main.current.camera);
      state.gl.setRenderTarget(null);
      state.scene.background = new Color(bgColor);
      FBOscene.background = new Color(bgColor);
    }
    material.current.uniforms.uTime.value = state.clock.getElapsedTime();
  });

  const shaderArgs = useMemo(
    () => ({
      uniforms: {
        uTexture: { value: target.texture },
        uRes: { value: { x: viewport.width, y: viewport.height } },
        uTime: { value: 0 },
        uPostProcessing: { value: postprocessing ? 1 : 0 },
      },
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        } 
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        uniform sampler2D uTexture;
        uniform float uTime;
        uniform vec2 uRes;
        uniform float uPostProcessing;

        void main() {
					vec2 uv = vUv;
					vec3 col = texture(uTexture, uv).rgb;

					if (uPostProcessing == 0.) {
						// No Postprocessing
						gl_FragColor = vec4(col, 1.0);	
					} else {
						// Cool postprocessing
						col += sin((uv.x + uv.y) * 1000.) * 0.1; // <- Dots
						col -= smoothstep(0.3, 1., length(uv - 0.5)); // <- Vignetting
						col += vec3(uv.x, uv.y, 1.); // <- Gradient
						gl_FragColor = vec4(col, 1.0);
					}
        } 
      `,
    }),
    [target.texture, viewport, postprocessing]
  );

  return (
    <>
      {createPortal(<Main ref={main} />, FBOscene)}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[viewport.width, viewport.height]} />
        <shaderMaterial ref={material} args={[shaderArgs]} />
      </mesh>
    </>
  );
};

export default FBO;
