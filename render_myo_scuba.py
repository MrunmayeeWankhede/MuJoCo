"""Render the physical, muscle-driven MyoSim scuba rollout."""
from myo_scuba import MyoScuba, ROOT
import mujoco
from PIL import Image


def main():
    sim = MyoScuba()
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 90, -5, 2.8
    camera.lookat[:] = [0, .1, 1]
    frames = []
    closeup = None
    left_closeup = None
    with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
        for step in range(round(3.5/sim.dt)):
            sim.step()
            if step >= round(1/sim.dt) and step % round(.04/sim.dt) == 0:
                renderer.update_scene(sim.data, camera=camera)
                frames.append(Image.fromarray(renderer.render().copy()))
                if closeup is None:
                    hand_camera = mujoco.MjvCamera()
                    hand_camera.azimuth = 105
                    hand_camera.elevation = -5
                    hand_camera.distance = .9
                    hand_camera.lookat[:] = [-.10, .05, 1.50]
                    options = mujoco.MjvOption()
                    options.tendongroup[:] = 0
                    renderer.update_scene(sim.data, camera=hand_camera, scene_option=options)
                    closeup = Image.fromarray(renderer.render().copy())
                    hand_camera.lookat[:] = sim.hand_position(sim.data, 'l')
                    hand_camera.distance = .6
                    hand_camera.azimuth = 90
                    renderer.update_scene(sim.data, camera=hand_camera, scene_option=options)
                    left_closeup = Image.fromarray(renderer.render().copy())
    output = ROOT/'results'
    output.mkdir(exist_ok=True)
    frames[0].save(output/'myo_scuba_preview.png')
    closeup.save(output/'myo_hand_preview.png')
    left_closeup.save(output/'myo_left_hand_preview.png')
    frames[0].save(output/'myo_scuba_preview.gif', save_all=True,
                   append_images=frames[1:], duration=40, loop=0)
    print(output/'myo_scuba_preview.gif')


if __name__=='__main__':main()
