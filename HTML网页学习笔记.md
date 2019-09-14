HTML网页学习笔记

1.网页的基本结构

```
<html>
	<head>
		<title></title>
	</head>
	<body>
	</body>
</html>
```

2.网页的字符集问题

如果网站中仅仅包含中文，那么推荐使用GB2312体积更小访问速度耿况

其他的我们可以使用UTF-8

设置字符集：

```
<head>
	<meta charset="utf-8"/>
</head>
```

要保证编码的文字与文件的编码方式相同

3.html的DTD文档声明

在html中写文件之前我们要首先进行DTD文档声明

声明方式

\<!DOCTYPE html>

4.基础标签学习

\<h>标签：用于给我们的文本添加标题语义，不能用于改变标题的样式

p标签告诉浏览器这是一个段落，在浏览器中会单独占一行

hr标签，为网页添加一条分割线，在浏览器中会单独占据一行。

\<!---->进行添加注释,webstorm快速插入注释ctrl+/

\<img src="" alt=“”>添加一张图片，alt中添加的内容提示浏览器，当图片找不到的时候，浏览器会显示什么不会独占一行，

br标签，进行换行（很少使用）

在选择路径的时候尽量选择相对路径，这样可移植性比较高，我们统一使用反斜杠/

a标签用于控制页面与页面之前的跳转

\<a href="">必须要有href属性

如果通过href指定url必须在地址前面加上http://或者https://

a标签href可以指定本地地址也可以指定网络地址，

a标签中有一个target属性，属性的作用就是用于控制如何跳转

\_self直接在当前的选项卡中，_black在一个新的选项卡中

title属性鼠标移动到标签上时会弹出的内容

base标签，统一控制当前网页上的所有的a标签的打开方式

```
<head>
<base target="“>
</head>
```

a标签的假链接, 点击后自动回到网页的顶部

利用a标签实现在页面中实现跳转,\<a href="#+id">直接跳转到id为id的位置，*滚动无动画

a标签可以跳转至网页的某个指定位置\<a href="url+#+Id">

列表属性

1.ul无序列表，ul与li标签一个组合，是一个整体，要结合在一起使用

2.ul标签中不推荐包含其他的标签

应用场景

1.新闻列表

2.商品列表

3.导航条

定义列表的格式

```
<dl>
	<dt></dt>
	<dd></dd>
	<dt></dt>
	<dd></dd>
<\dl>
```

dt相当于标题，dd相当于对于标题的描述

定义列表的应用：

1.网页底部的相关信息制作

2.做图文混排

注意点：和ul ol一样 dl和dt dd是一个整体所以他们一般情况下不会单独出现，都是一起出现

一个dt对应一个dd

当需要丰富界面的时候在dd标签中再添加别的标签

表格标签

格式

```
<table>
	<tr>
		<td></td>
	</tr>	
</table>
```

tr标签代表表格中的一行数据，td标签代表一行中的一个单元格

表单元素：用来收集用户信息

格式

```
<form>
	<!--明文输入框-->
	<input type="text">
	<!--暗文输入框-->
	<input type="password">
	<!--设置值-->
	<input type="text" value="">
	<!--单选框-->
	<input type="radio">男
	<!--单选框互斥-->
	<input type="radio" name="gender">男
	<input type="radio" name="gender">女
	<!--设置默认选中-->
	<input type="radio" name="gender" checked=”checked">保密
	<!--多选框-->
	<input type="checkbox">
	<!--按钮-->
	<input type="button" value="男">
	<!--使用图片定义一个按钮-->
	<input type="image">
	<!--
		清空表单的数据
		有默认的标题，但可以进行修改
	-->
	<input type="reset">
	<!--提交数据-->
	<input type="submit">
	<!--通过form action告诉提交到那个服务器-->
</form>
```

```
<form>
<label for="account">账号：</label><input type="text" id="account">
<label for="pwd">密码：</label><input type="passward" id="pwd">
</form>
```

使得文字与输入框绑定起来，在我们点击文字的时候，输入框同时也会进行聚焦

datalist标签

给输入框绑定带选项

格式

```html
<input type="text" list="test">
<datalist id="test">
    <option>待选</option>
    <option>待选</option>
    <option>待选</option>
    <option>待选</option>
    <option>待选</option>
</datalist>
```

select标签下拉列表

```html
<select>
	<option>标签数据</option>
	<!--设置默认选中-->
	<option selected="selected"><option>
	<!--设置在下拉列表中进行分组-->
	<optgroup label="分组名称">
		<option></option>
		<option></option>
		<option></option>
		<option></option>
		<option></option>
	</optgroup>
</select>
```

textarea定义一个多行的输入框

```
<textarea cols="" rows="">指定宽度和宽度
<!--有默认的宽度和高度-->
</textarea>
```

video标签播放视频

\<video src=“”>\</video>默认情况下不会自动播放，需要添加autoplay属性

src：指定视频的地址

autoplay:指定视频自动播放

controls：给视频添加控制条

poster：给视频添加一张封面

loop：进行循环播放

preload：进行预加载preload与autoplay属性相互冲突

muted：设置视频为静音

第二个视频标签的格式目的是为了解决不同浏览器对视频的适配问题

```
<video>
	<source src="" type=""></source>
	<source src="" type=""></source>
</video>
```

音频标签:audio

格式：

```
<audito src="">
</audio>
<audio>
	<source src="" type=""></source>
</audio>
```

基本与video标签一样

详情和概要标签：

```
<details>
	<summary></summary>
</details>
```

解决空间问题，将details中的内容隐藏起来，当打开的时候只显示summary中的内容

marquee标签：跑马灯效果

```
<marquee></marquee>
<!--direction控制方向-->
<!--scrollamount控制速度-->
<!--loop控制循环次数-->
<!--behavior
:slide滚到边界停止
:alternate滚动到边界就弹回
-->
```

