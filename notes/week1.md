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


总结：
1、ManiSkill是一个机器人仿真环境
2、env_id="PushCube-v1"表示推方块任务
3、PPO是一种强化学习算法
4、total_timesteps控制训练总步数，但运行慢不一定只由它决定
5、evaluation/video rendering 也会很慢
6、Colab重启后，/content里的文件可能会丢，需要重新安装或更新cd到目录

可以简单理解：
env.reset()：游戏重新开始
env.step(action)：机器人做一个动作，环境往前走一步
observation/obs：机器人看到/知道的状态
action：机器人执行的动作
reward：这一步做得好不好
done/terminated/truncated：这一局是否结束
policy：机器人当前的决策策略
PPO：训练policy的一种方法
evalution：测试现在训练得怎么样

env.reset()会初始化一个episode
env.step(action)会让机器人执行一个动作
action是策略网络输出的动作
obs是环境返回给机器人的观察
reward用来告诉算法这一步做得好不好
PPO不断用reward来更新policy
eval阶段只是测试，不是训练
capture_video会保存视频，因此可能变慢
Colab重启后工作目录会丢，要重新安装或cd


1、PushCube-v1这个任务机器人做什么？推方块任务，机械臂将立方体推到目标点位。6自由度桌面机械臂，在仿真平面内推动立方体方块，目标是把方块推到指定标记的目标坐标点位；任务奖励会根据方块与目标的距离设计稠密奖励，完成到位即判定任务成功。分为两种训练方案：低维状态输入（读取方块、机械臂坐标）、视觉RGB图像输入（ppo_rgb，仅靠相机画面完成推方块）
2、PPO是在学什么？PPO（近端策略优化）是强化学习算法
3、为什么我调小total_timesteps后还是可能很慢？total_timesteps只是全局总终止步数，只控制训练什么时候结束，不会改变单次迭代的计算耗时
4、Colab重启后为什么会找不到pop.py？Colab的运行时是临时容器，重启/断开连接后容器会完全重置，本地临时文件全部清空
5、最不理解的3个词是什么？
·GAE广义优势估计：PPO用来衡量“当前动作比平均动作好多少”的计算方法，结合时序差分与蒙特卡洛回报平滑优势；新手容易分不清优势函数、价值、奖励三者区别，是PPO损失函数的核心前置计算
·trajectory轨迹（episode）：一条完整交互流程：从环境重置开始，机械臂持续执行动作直到任务成功/达到最大步数，完整保存每一步观测、动作、奖励、物理状态的时序数据；专家演示h5文件、RL训练采集样本全部都是轨迹数据，容易混淆单步transition与完整episode轨迹
·clip裁剪目标（PPO核心）：PPO独有的约束机制，通过限制新旧策略动作概率比值区间，防止单次梯度更新幅度太大导致策略崩溃；公式抽象，新手很难理解为什么需要裁剪、裁剪区间的作用

-p:目录不存在则创建，存在也不报错
wget -q:静默下载，不打印多余日志
gymnasium as gym:新版Gym仿真交互标准库，提供gym.make()创建环境、reset/step交互接口
env.action_space.sample():随机动作采样，生成机械臂随机关节控制指令
env.step(action)标准5元组返回（Gymnasium新版规范）：
      ·obs：执行动作后相机、机器人传感器观测
      ·rew：单步奖励值，插销任务越靠近目标奖励越高
      ·terminated：任务成功终止（插销完全插入孔内，自然结束）
      ·truncated：步数截断终止（达到环境最大仿真步数，强制结束）
      ·info：调试信息字典，包含elapsed_steps当前已运行总步数
.item()：把GPU张量转为普通Python数字
env.render()：返回批量GPU张量，shape为(1,H,W,3)，1是批量纬度，图像存在CUDA显存上
并行仿真FPS统计标准算法：总仿真帧=环境数量*单环境步数
pd_joint_delta_pos：PD关节增量位置控制，最稳定的机械臂关节控制方案；还可切换末端6D力控、爪控等模式
dense：稠密奖励，每一步都给小奖励，训练收敛更快
sparse：稀疏奖励，仅任务成功时给1，其余为0，难度更高

PPO全称Proximal Policy Optimization（近端策略优化），是目前机器人强化学习最主流、工程最稳定的策略梯度类RL算法，ManiSkill里用它训练机械臂、四足机器人完成抓取、插销、堆叠等任务。分为两个版本：1、ppo.py：低纬观测（关节向量、目标位姿）训练；2、ppo_rgb.py：视觉端到端训练（输入RGB图像/点云，纯视觉无人工状态）
PPO用简单裁剪目标函数限制新旧策略差异，兼顾稳定、易实现、训练效率
PPO核心逻辑（分4大模块）
1、环境交互：收集批量仿真轨迹（ManiSkill GPU并行采集）
<img width="741" height="134" alt="image" src="https://github.com/user-attachments/assets/7269b06d-cf89-438a-b7e3-5eeee6b3547e" />

2、优势函数计算（GAE广义优势估计，ManiSkill默认使用）
<img width="406" height="279" alt="image" src="https://github.com/user-attachments/assets/393f5f6b-be7d-42f1-a4d7-d974e6c7fb80" />

3、核心裁剪损失函数PPO-Clip（最关键公式）
<img width="623" height="238" alt="image" src="https://github.com/user-attachments/assets/e4db284c-6e12-41a3-98e8-cb228aa2c70b" />

4、联合总损失（策略损失+价值损失+熵正则）
<img width="735" height="270" alt="image" src="https://github.com/user-attachments/assets/3f8c0ee3-c404-45c8-b2c7-911eebe70069" />

5、迭代更新流程（循环往复直到收敛）
<img width="598" height="205" alt="image" src="https://github.com/user-attachments/assets/65bd6d30-0cb8-4e97-aa65-2778d5180a4f" />

PPO训练耗时很久，无法仅凭打印文字判断训练好坏；TensorBoard用可视化曲线直观监控收敛情况：
1、奖励稳步上升：训练正常
2、奖励震荡/断崖下跌：超参数不合适、策略更新过大
3、熵持续归零：策略过早固化，缺少探索，容易卡在局部最优

ManiSkillTrajectory Dateset：代码内程序化读取轨迹数据，用于算法训练、数据分析
replay——trajectory：命令行可视化回放，生成视频直观查看动作，不提供程序可读取的结构化数据

index_dict：多层嵌套字典/Group批量时序索引工具
dict_to_list_of_dicts：配套工具，把（T，dim）时序字典转为长度为T的字典列表（每一项时单步状态）

主线流程：
创建环境->reset初始化->policy产生action->env.step(action)->得到obs/reward/done->训练或评估

能复现的PushCube smoke test命令（能确认环境坏没坏、ManiSkill能启动、PushCube-v1能运行、PPO脚本能执行、结果目录能生成）：


~~~
python ppo.py \
      --env_id="PushCube-v1" \
      --exp_name="smoke-pushcube" \
      --num_envs=32 \
      --num_eval_envs=1 \
      --num_eval_steps=5 \
      --num_steps=20 \
      --update_epochs=2 \
      --num_minibatches=4 \
      --total_timesteps=12800 \
      --eval_freq=1 \
      --no-capture-video

~~~
#安装ManiSkill依赖与本体
!pip install mani-skill
#安装仿真物理引擎配套依赖
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

~~~
import gymnasium as gym
import mani_skill.envs
env=gym.make("PushCube-v1",obs_mode="state",control_mode="pd_ee_delta_pose")
obs,info=env.reset()
print(type(obs))
print(obs)
action=env.action_space.sample()
obs,reward,terminated,truncated,info=env.step(action)
print(action)
print(reward)
print(terminated,truncated)

gymnasium：是OpenAIGym的官方继任版本，现在强化学习环境统一用这个库，提供标准环境交互接口（make/reset/step)
mani_skill：安装的机器人仿真库，内置PushCube、PickCube等机械臂操作任务
mani_skill.envs：专门存放所有仿真任务环境
gym.make()：创建仿真环境实例，返回一个环境对象env
"PushCube-v1"：任务ID，代表机械臂推方块仿真任务，v1是版本号
obs_mode="state"：观测模式。state只输出纯数字状态向量（关节角度、物体坐标等），不需要图像渲染，速度快，适合PPO等算法训练；rgb模式会输出相机图片，训练速度慢
control_mode="pd_ee_delta_pose"：机械臂控制模式。
pd：PD位置闭环控制器
ee：end-effector末端执行器（机械爪）
delta_pose：增量位姿控制，每次动作之修改末端相对上一帧的偏移
env.reset()：重置环境到初始随机场景（方块随机摆放、机械臂回到初始位置），每一轮episode开始必须调用
obs：初始观测值（包含机器人、方块全部状态信息）
info：附加信息字段（场景参数、碰撞标记、物体坐标等辅助数据）
ManiSkill默认用GPU加速，所有观测、动作、奖励全是PyTorch张量，方便直接送入神经网络训练，不用手动转数组
env.action_space：环境的动作空间，定义机械臂能输出的动作纬度、取值范围
.sample()：从动作空间里随机采样一个合法动作（随机给机械臂末端一个微小偏移指令）
action：保存随机动作tensor，用来传给环境执行一步仿真
terminated：任务是否成功终止（方块推到目标区域则为True）
truncated：回合是否超时阶段（步数达到最大限制，提前结束）
info：额外调试信息

