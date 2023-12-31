# MPM_YarnClothSim
MPM yarn based cloth simulation

欢迎来玩 MPM yarn based cloth simulation， 这是基于蒋陈凡夫老师的 Anisotropic elastoplasicity for cloth, knit and hair frictional contact 写的 模拟 yarn based cloth 的程序（目前只能模拟yarn和yarn based cloth，不能模拟沙子或三角网格布料orz），参考了知乎橘子猫大佬的2D版本（https://zhuanlan.zhihu.com/p/414356129） （把大佬的2D版本改成了3D版本orz）
由于本人很菜QAQ，程序目前存在如下问题：
- return mapping好像实现错了，所以在模拟中没有执行这个步骤；
- 出现一些奇怪的现象。

如果您能帮助我改进，我将万分感谢！

## Usage

运行方法：

- 用 Blender2.93 打开 MPM_YarnBasedClothSim.blend（主要的模拟程序是 UnitTest_AEPMPM.py (由于改文件名会发生错误，所以我没有改QAQ）,其他的一些比较次要的工具代码是嵌入在blender中的文本）；
- 安装 taichi-Blend 插件；
- 切换到 MyPanel 文本并运行（此时视图中出现交互界面）；

创建yarn based cloth：

- 针织：创建物体->细分->create Stitch for Obj (如果面的index是按顺序的，可以在物体上直接转换，如果面的index是乱序的，可以在编辑模式下转换每一个面)（这里是根据Yuksel等人的Stitch Mesh实现的）(文件中的Plane.001是针织布料的一个例子，钉固了右边一列的顶点，并且钉固顶点在模拟的过程中缓慢向右移动)
- 编织：创建物体->细分->wove face
- 导入：交互界面的load BCC file 按钮。

进行模拟：

- 在 Blender 2.93 目录(也就是blender.exe所在的目录)下创建路径
  - work_dir2/frames
- 钉固顶点：编辑曲线并选中要钉固的顶点(如果没有钉固顶点，需要创建一个浮空的顶点并选中它，因为传递给 taichi 类的张量不能为空（原谅我暂时没有找到更好的解决办法orz），注意不要选中所有的顶点，这样就不能动了。)
- 球体/地面碰撞：在GridCollision中修改球体的位置和半径或地面高度
- 运行：点击要模拟的物体（例如Plane.001），切换到UnitTest_AEPMEM,点击运行
- 运行完成后导入动画：点击 load Animation 按钮 (动画在运行的过程中会写到 work_dir2/frames 下，点击此按钮后保存)
