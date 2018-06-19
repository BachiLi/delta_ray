import numpy as np
import image
import camera
import delta_ray

resolution = [256, 256]
cam = camera.Camera(position   = np.array([278, 273, -800], dtype = np.float32),
                    look_at    = np.array([278, 273, -799], dtype = np.float32),
                    up         = np.array([  0,   1,    0], dtype = np.float32),
                    fov        = 39.3077,
                    clip_near  = 0.01,
                    clip_far   = 10000.0,
                    resolution = resolution)

cam = delta_ray.Camera(cam.cam_to_world,
                       cam.world_to_cam,
                       cam.sample_to_cam,
                       cam.cam_to_sample)

# yellow = delta_ray.Material(np.array([0.8, 0.8, 0.3], dtype = np.float32))
# pink = delta_ray.Material(np.array([0.7, 0.3, 0.7], dtype = np.float32))
grey = delta_ray.Material(np.array([0.75, 0.75, 0.75], dtype = np.float32))
green = delta_ray.Material(np.array([0.14, 0.48, 0.1], dtype = np.float32))
red = delta_ray.Material(np.array([0.64, 0.06, 0.06], dtype = np.float32))
black = delta_ray.Material(np.array([0, 0, 0], dtype = np.float32))

# tri_0 = delta_ray.Shape(np.array([[-1.7, 1.0,  0.0], [1.0, 1.0,  0.0], [-0.5, -1.0,  0.0]], dtype = np.float32),
#                         np.array([[0, 1, 2]], dtype = np.int32),
#                         yellow)
# tri_1 = delta_ray.Shape(np.array([[-1.2, 0.0, -1.5], [1.5, 0.0,  1.0], [-0.5, -1.5, -0.5]], dtype = np.float32),
#                         np.array([[0, 1, 2]], dtype = np.int32),
#                         grey)
# tri_2 = delta_ray.Shape(np.array([[-0.0, 1.4,  0.5], [2.0, 1.4,  0.5], [ 1.0, -0.6,  0.0]], dtype = np.float32),
#                         np.array([[0, 1, 2]], dtype = np.int32),
#                         pink)
# shapes = [tri_0, tri_1, tri_2]

# floor = delta_ray.Shape(np.array([[-1.5, 0.0, -1.5],
#                                   [-1.5, 0.0,  1.5],
#                                   [ 1.5, 0.0, -1.5],
#                                   [ 1.5, 0.0,  1.5]], dtype = np.float32),
#                         np.array([[0, 1, 2], [1, 2, 3]], dtype = np.int32),
#                         grey,
#                         None)
# blocker = delta_ray.Shape(np.array([[-0.4, 0.5, -0.4],
#                                     [-0.4, 0.5,  0.4],
#                                     [ 0.2, 0.5, -0.4],
#                                     [ 0.5, 0.5,  0.4]], dtype = np.float32),
#                           np.array([[0, 1, 2], [1, 2, 3]], dtype = np.int32),
#                           grey,
#                           None)
# light_mesh = delta_ray.Shape(np.array([[-0.5, 1.0, -0.5],
#                                        [-0.5, 1.0,  0.5],
#                                        [ 0.5, 1.0, -0.5],
#                                        [ 0.5, 1.0,  0.5]], dtype = np.float32),
#                         np.array([[0, 1, 2], [1, 2, 3]], dtype = np.int32),
#                         grey,
#                         None)
# shapes = [floor, blocker, light_mesh]

floor = delta_ray.Shape(np.array([[552.8, 0.0,   0.0],
                                  [  0.0, 0.0,   0.0],
                                  [  0.0, 0.0, 559.2],
                                  [549.6, 0.0, 559.2]], dtype = np.float32),
                        np.array([[0, 1, 3], [2, 3, 1]], dtype = np.int32),
                        grey,
                        None)

ceiling = delta_ray.Shape(np.array([[556.0, 548.8,   0.0],
                                    [556.0, 548.8, 559.2],
                                    [  0.0, 548.8, 559.2],
                                    [  0.0, 548.8,   0.0]], dtype = np.float32),
                          np.array([[0, 1, 3], [2, 3, 1]], dtype = np.int32),
                          grey,
                          None)

back_wall = delta_ray.Shape(np.array([[549.6,   0.0, 559.2],
                                      [  0.0,   0.0, 559.2],
                                      [  0.0, 548.8, 559.2],
                                      [556.0, 548.8, 559.2]], dtype = np.float32),
                            np.array([[0, 1, 3], [2, 3, 1]], dtype = np.int32),
                            grey,
                            None)

green_wall = delta_ray.Shape(np.array([[  0.0,   0.0, 559.2],
                                       [  0.0,   0.0,   0.0],
                                       [  0.0, 548.8,   0.0],
                                       [  0.0, 548.8, 559.2]], dtype = np.float32),
                             np.array([[0, 1, 3], [2, 3, 1]], dtype = np.int32),
                             green,
                             None)

red_wall = delta_ray.Shape(np.array([[552.8,   0.0,   0.0],
                                     [549.6,   0.0, 559.2],
                                     [556.0, 548.8, 559.2],
                                     [556.0, 548.8,   0.0]], dtype = np.float32),
                           np.array([[0, 1, 3], [2, 3, 1]], dtype = np.int32),
                           red,
                           None)

large_box = delta_ray.Shape(np.array([[423.0, 330.0, 247.0],
                                      [265.0, 330.0, 296.0],
                                      [314.0, 330.0, 456.0],
                                      [472.0, 330.0, 406.0],
                                      [423.0,   0.0, 247.0],
                                      [472.0,   0.0, 406.0], 
                                      [314.0,   0.0, 456.0],
                                      [265.0,   0.0, 296.0]],
                                      dtype = np.float32),
                            np.array([[ 0,  1,  2],
                                      [ 0,  2,  3],
                                      [ 4,  0,  3],
                                      [ 4,  3,  5],
                                      [ 5,  3,  2],
                                      [ 5,  2,  6],
                                      [ 6,  2,  1],
                                      [ 6,  1,  7],
                                      [ 7,  1,  0],
                                      [ 7,  0,  4],
                                      [ 5,  6,  7],
                                      [ 5,  7,  4]], dtype = np.int32),
                            grey,
                            None)

# occluder = delta_ray.Shape(np.array([[423.0, 330.0, 247.0],
#                                      [265.0, 330.0, 296.0],
#                                      [314.0, 330.0, 456.0],
#                                      [472.0, 330.0, 406.0]],
#                                       dtype = np.float32),
#                             np.array([[ 0,  1,  2],
#                                       [ 0,  2,  3]], dtype = np.int32),
#                             grey,
#                             None)

occluder = delta_ray.Shape(np.array([[400.0, 300.0, 250.0],
                                     [400.0, 300.0, 400.0],
                                     [250.0, 300.0, 400.0],
                                     [250.0, 300.0, 250.0]],
                                      dtype = np.float32),
                            np.array([[ 0,  1,  2],
                                      [ 0,  2,  3]], dtype = np.int32),
                            grey,
                            None)

small_box = delta_ray.Shape(np.array([[130.0, 165.0,  65.0],
                                      [ 82.0, 165.0, 225.0],
                                      [240.0, 165.0, 272.0],
                                      [290.0, 165.0, 114.0],
                                      [290.0,   0.0, 114.0],
                                      [240.0,   0.0, 272.0],
                                      [130.0,   0.0,  65.0],
                                      [ 82.0,   0.0, 225.0]],
                                      dtype = np.float32),
                            np.array([[ 0,  1,  2],
                                      [ 0,  2,  3],
                                      [ 4,  3,  2],
                                      [ 4,  2,  5],
                                      [ 6,  0,  3],
                                      [ 6,  3,  4],
                                      [ 7,  1,  0],
                                      [ 7,  0,  6],
                                      [ 5,  2,  1],
                                      [ 5,  1,  7],
                                      [ 4,  5,  7],
                                      [ 4,  7,  6]], dtype = np.int32),
                            grey,
                            None)

light_mesh = delta_ray.Shape(np.array([[343, 548.3, 227],
                                       [343, 548.3, 332],
                                       [213, 548.3, 332],
                                       [213, 548.3, 227]], dtype = np.float32),
                             np.array([[0, 1, 3], [2, 3, 1]], dtype = np.int32),
                             black,
                             None)

# shapes = [floor, ceiling, back_wall, green_wall, red_wall, large_box, small_box, light_mesh]
shapes = [floor, occluder, light_mesh]

# materials = [yellow, grey, pink]
materials = [grey, green, red]

light = delta_ray.Light(light_mesh,
                        np.array([18.4, 15.6, 8.0], dtype = np.float32))
light_mesh.light = light
lights = [light]

# result = delta_ray.render(cam, shapes, materials, lights, resolution, 4)
# img = result.image
# dx_img = result.dx_image
# dy_img = result.dy_image
# image.imwrite(img, 'results/image.exr')
# image.imwrite(dx_img, 'results/dx_image.exr')
# image.imwrite(dy_img, 'results/dy_image.exr')

delta_ray.test_render(cam, shapes, materials, lights, resolution, 4)
