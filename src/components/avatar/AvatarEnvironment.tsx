export function AvatarEnvironment() {
  return (
    <>
      <ambientLight intensity={0.42} color="#dff8ff" />
      <directionalLight position={[0.8, 2.4, 2.2]} intensity={1.9} color="#ffffff" />
      <spotLight position={[-2.3, 2.1, -1.4]} intensity={2.1} angle={0.42} penumbra={1} color="#38bdf8" />
      <spotLight position={[2.4, 2.3, -1.3]} intensity={1.45} angle={0.45} penumbra={1} color="#818cf8" />
      <pointLight position={[0, 0.4, 1.4]} intensity={0.72} color="#dff8ff" />
      <fog attach="fog" args={['#0f172a', 8, 26]} />
      <color attach="background" args={['#0f172a']} />
    </>
  );
}
