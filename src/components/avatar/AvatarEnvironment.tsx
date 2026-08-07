export function AvatarEnvironment() {
  return (
    <>
      <ambientLight intensity={0.42} color="#dff8ff" />
      <directionalLight position={[0.8, 2.4, 2.2]} intensity={1.9} color="#ffffff" />
      <spotLight position={[-2.3, 2.1, -1.4]} intensity={2.1} angle={0.42} penumbra={1} color="#58e6ff" />
      <spotLight position={[2.4, 2.3, -1.3]} intensity={1.45} angle={0.45} penumbra={1} color="#9f7aea" />
      <pointLight position={[0, 0.4, 1.4]} intensity={0.72} color="#dff8ff" />
      <fog attach="fog" args={['#040711', 8, 26]} />
      <color attach="background" args={['#040711']} />
    </>
  );
}
