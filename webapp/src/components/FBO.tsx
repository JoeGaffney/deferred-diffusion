import { useMemo, useRef, useState } from "react";
import { useThree, createPortal, useFrame, extend } from "@react-three/fiber";
import { useFBO } from "@react-three/drei";
import { Scene, Color, WebGLRenderer, RGBAFormat, UnsignedByteType } from "three";
import { button, useControls } from "leva";
import Main from "./Main";
import { io } from "socket.io-client";

const socket = io("http://localhost:5000"); // Flask backend address

const FBO = () => {
  const { viewport } = useThree();
  const material = useRef();
  const main = useRef();
  const [capture, setCapture] = useState(false);

  const { postprocessing, bgColor } = useControls({
    postprocessing: true,
    bgColor: "#000000",
    testColor: "#000000",
    number: 3,
    sendTarget: button(() => setCapture(true)),
  });

  const FBOscene = useMemo(() => {
    const FBOscene = new Scene();
    return FBOscene;
  }, []);
  const target = useFBO({
    format: RGBAFormat,
    type: UnsignedByteType,
  });

  const captureRenderBuffer = (state, target) => {
    console.log("Processing texture.", target);

    const width = target.texture.image.width;
    const height = target.texture.image.height;
    const imageData = new ImageData(width, height);
    state.gl.readRenderTargetPixels(target, 0, 0, width, height, imageData.data);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    ctx.putImageData(imageData, 0, 0);
    console.log("Canvas imageData.", imageData);

    const base64Image = canvas.toDataURL("image/png");
    // Send the Base64 image data over the socket
    socket.emit("rendered_image", { data: base64Image });
    console.log("Base64 image sent to server.", base64Image);
  };

  useFrame((state) => {
    if (main.current.camera) {
      state.gl.setRenderTarget(target);
      state.gl.render(FBOscene, main.current.camera);
      state.gl.setRenderTarget(null);

      if (capture) {
        setCapture(false);
        captureRenderBuffer(state, target);
      }

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
