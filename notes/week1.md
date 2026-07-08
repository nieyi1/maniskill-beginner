报错1:GPU张量无法直接转numpy

<img width="692" height="398" alt="10e68095efc34a709095061586288ecd" src="https://github.com/user-attachments/assets/fecf88e1-7103-4ff2-9ce0-70d233666fab" />

代码：
import matplotlib.pyplot as plt
env = gym.make("PegInsertionSide-v1", render_mode="rgb_array")
env.reset()
plt.imshow(env.render()[0].numpy()) # we take [0].numpy() as everything is a batched tensor

出错代码：plt.imshow(env.render()[0].numpy())

报错：
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_3791/ 4269541533.py  in <cell line: 0>()
      2 env = gym.make("PegInsertionSide-v1", render_mode="rgb_array")
      3 env.reset()
----> 4 plt.imshow(env.render()[0].numpy()) # we take [0].numpy() as everything is a batched tensor

TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

问题根源：仿真图像存在GPU显存，numpy仅支持CPU内存数据

意思是：拿到的图像张量存在GPU(cuda:0)上，numpy只能读取CPU内存的数据，不能直接操作GPU张量，必须先把张量从GPU传到CPU

env.render()返回的图片是CUDA GPU张量

直接调用.numpy()会报错，numpy不支持GPU上的数据

官方注释只写了[0].numpy()，漏了.cpu()转移设备，所以出错

所以把代码改为：plt.imshow(env.render()[0].cpu().numpy())即可

新增.cpu()，作用：把GPU上的图像数据拷贝到CPU内存，之后才能转numpy数组给matplotlib绘图

<img width="661" height="573" alt="image" src="https://github.com/user-attachments/assets/e2ea0573-2171-4864-a05a-98f941c63ff6" />

报错2:GPU PhysX重复初始化报错

<img width="1080" height="725" alt="image" src="https://github.com/user-attachments/assets/dce58efa-07a8-4aa8-b6d6-c129b812f32d" />

报错信息：RuntimeError：GPU PhysX can only be enabled once before any other code involving PhysX

出错代码：env=gym.make("PickCube-v1",num_envs=num_envs)

问题根源：1、先后运行了多段ManiSkill仿真单元格，底层PhysX GPU物理引擎被重复初始化2、Colab内核未重启，全局PhysX状态残留，不支持二次开启GPU模式

解决方案：1、执行【代码执行程序->重启会话】清空运行状态2、重启后直接运行本段代码，不要先执行其他仿真单元格

<img width="987" height="627" alt="image" src="https://github.com/user-attachments/assets/c828df25-44aa-4b27-b7bc-3a8038e617c7" />

报错3:包装器无render_rgb_array属性

<img width="1078" height="720" alt="image" src="https://github.com/user-attachments/assets/19a4ebd0-d8fd-46f8-9b35-47834546265c" />

<img width="1028" height="680" alt="image" src="https://github.com/user-attachments/assets/3af4875f-224f-4750-a197-d6ba94d9585e" />

报错信息：AttributeError: 'TimeLimitWrapper' object has no attribute 'render_rgb_array'

出错代码：rgbs=env.render_rgb_array()

问题根源：gym创建环境会自动包裹TimeLimitWrapper时长限制封装层，render_rgb_array()属于底层原生仿真环境的方法，外层包装器不提供该接口、

解决方案：通过env.unwrapped剥离所有包装层，获取原始环境对象，修改为rgbs=env.unwrapped.render_rgb_array()

<img width="1073" height="708" alt="image" src="https://github.com/user-attachments/assets/4635778c-9a15-4acc-b4fd-6a2058cee6f2" />

<img width="928" height="718" alt="image" src="https://github.com/user-attachments/assets/5ea394b1-3b11-41dd-893e-64ebb450f1cd" />

报错4:KeyError:'pointcloud'

<img width="1101" height="427" alt="image" src="https://github.com/user-attachments/assets/5c7d3dbf-74e6-4ce6-909c-94fe09f27026" />

报错信息：KeyError:'pointcloud'

问题根源：当前环境obs_mode不是pointcloud，仿真返回的观测字典obs没有三维点云数据字段，函数读取不存在的键触发报错；单元格注释明确要求必须使用obs_mode="pointcloud"

解决方案：上方交互面板将obs_mode下拉选项切换为pointcloud，然后重新执行环境创建与env.reset()，生成带点云数据的观测。然后再次运行show_pointcloud(obs)绘制三维点云

<img width="1104" height="622" alt="image" src="https://github.com/user-attachments/assets/5521b5ea-f6a3-47b6-8041-839d7cbb2dae" />

补充知识点：

obs_mode是创建环境的核心参数，决定机器人能“看到”什么数据：1、rgb：仅2D彩色图像；2、rgb+depth+segmentation：彩色+深度+物体分割；3、pointcloud：三维空间点云（包含每个像素的XYZ三维坐标，适合3D视觉机器人算法）

报错5:回放函数render_rgb_array属性缺失

<img width="1103" height="622" alt="image" src="https://github.com/user-attachments/assets/f37328b1-583b-443a-b80b-7faf8b6324d1" />

报错信息：AttributeError:'TimeLimitWrapper' object has no attribute 'render_rgb_array'

出错位置：replay()函数内部env.render_rgb_array()

问题根源：gym自动封装TimeLimitWrapper时长限制包装器，render_rgb_array属于底层原生环境接口，外层包装对象无法直接调用

解决方案：修改函数内渲染代码，使用env.unwrapped.render_rgb_array()剥离包装层获取原生环境

只要调用ManiSkill底层专属渲染/仿真接口(render_rgb_array、print_sim_details)，都必须加.unwrapped解包环境

只要env被任意包装器(TimeLimitWrapper、RecordEpisode等）包裹后：原生环境自带的.agent/.render_rgb_array()/.set_array_dict()都不能直接env.xxx；必须写成env.unwrapped.xxx，先剥离所有外层封装拿到底层仿真环境

<img width="1053" height="628" alt="image" src="https://github.com/user-attachments/assets/7dd2250a-8515-4b00-9db3-260eff3e628d" />

报错6:命令行不识别参数--num-procs

<img width="1095" height="280" alt="image" src="https://github.com/user-attachments/assets/260532c4-4d50-489e-97ef-9edd661c2906" />

报错：Unrecognized options:--num-procs

原因：replay_trajectory.py脚本无--num-procs启动参数，参数已废弃

解决：删除命令里--num-procs 1片段再执行

<img width="1031" height="213" alt="image" src="https://github.com/user-attachments/assets/c955fd94-e094-4770-bd7c-796be2f3f823" />

报错7:RecordEpisode包装器无agent属性

<img width="1091" height="638" alt="image" src="https://github.com/user-attachments/assets/d19caa6a-2dd1-4ba2-94eb-62992cffdbbd" />

报错信息：AttributeError:'RecordEpisode' object has no attribute 'agent'

出错代码：print(env.agent.keyframes.keys())

问题根源：RecordEpisode是视频录制包装器，包裹了原生仿真环境；.agent属于底层原生环境的属性，外层包装对象无法直接访问

解决方案：通过env.unwrapped剥离包装层，修改为print(env.unwrapped.agent.keyframes.keys())

<img width="942" height="288" alt="image" src="https://github.com/user-attachments/assets/085d99ee-0eb6-4565-ba7e-72b934f1e8c7" />










