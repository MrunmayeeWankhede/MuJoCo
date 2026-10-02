"""Save a still and animated scuba preview; requires an OpenGL context."""
import mujoco
from PIL import Image
from scuba import ScubaEnv, ROOT


def main():
    env = ScubaEnv()
    env.reset()
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 20, -10, 3.4
    camera.lookat[:] = [0, 0, 1.15]
    frames = []
    output = ROOT/'results'/'scuba'
    output.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
        for step in range(200):
            env.step(env.scripted_action())
            if step % 2 == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(Image.fromarray(renderer.render().copy()))
        frames[15].save(output/'scuba_preview.png')
        frames[0].save(output/'scuba_preview.gif', save_all=True,
                       append_images=frames[1:], duration=40, loop=0)
    print(output/'scuba_preview.gif')


if __name__ == '__main__':
    main()
