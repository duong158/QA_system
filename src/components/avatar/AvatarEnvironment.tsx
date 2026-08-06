export function AvatarEnvironment() {
  return (
    <>
      <ambientLight intensity={1.2} color="#9adfff" />
      <spotLight position={[3, 7, 5]} intensity={4.4} angle={0.35} penumbra={1} color="#ffffff" castShadow />
      <spotLight position={[-4, 5, -1]} intensity={2.8} angle={0.42} penumbra={1} color="#9f7aea" />
      <pointLight position={[0, -1, 3]} intensity={1.4} color="#58e6ff" />
      <fog attach="fog" args={['#040711', 8, 26]} />
      <color attach="background" args={['#040711']} />
    </>
  );
}