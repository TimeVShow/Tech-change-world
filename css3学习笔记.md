# css3学习笔记

<a href="#4">test</a>

## <h2 id="1">过渡模块</h2>

transition-property:+属性：声明哪一个属性需要动画过渡

transition-duration:+时间：声明过渡动画的时间

transition-delay:+时间：延迟多少秒之后开始动画

transition-timing-function:控制动画进行的速度

<span style="color:red">*</span>当需要用到过渡属性的时候，二者缺一不可

<span style="color:red">*</span>属性必须要发生改变

<span style="color:red">*</span>当有多个属性需要同时执行过渡时用,隔开

基本套路：首先搭建好基本的框架，再修改要修改的元素，最后添加上过渡的声明。

## <h2 id="2"> 2D转换模块</h2>

transform:rotate(X+deg);旋转X度

transform:translate(x,y);水平平移x上下平移y

transform:scale(x,y)水平方向缩放x，垂直方向缩放y

水平垂直都一样那么就只用写一个

默认情况下都是以自己中心为中心进行转换

transform-origin：设置旋转中心的位置。

transform:rotateX(X+deg)围绕X轴转 rotateY(X+deg)围绕Y轴转 

<h2>近大远小效果</h2>

perspective:设置近大远小效果

<span style="color:red">*</span>要在父元素中进行设置

<span style="color:red">*</span>左右布局水平居中，设置元素为行内块级元素，在设置上下外边距进行调整即可。

<h2 id = "3">阴影</h2>

1.给盒子添加阴影

box-shadow:水平偏移，垂直偏移，模糊度，阴影拓展，阴影颜色 内外阴影

阴影拓展在阴影外边添加额外的阴影

快速添加只需指定水平偏移以及垂直偏移，模糊度阴影默认的颜色为跟随内容的颜色

2.给文字添加阴影

text-shadow:水平偏移，垂直偏移，模糊度，阴影颜色 

## <h2 id = “4">动画</h2>

<span style="color:red">*</span>过渡动画需要人为去触发，才能显示

<span style="color:red">*</span>动画则不需要主动去触发。

<span style="color:red">*</span>动画三属性：

1.animation-name:动画名字

2.动画的声明

```html
@keyframes lnj{
	from{
		声明动画之前的样式
	}
	to{
		声明动画之后的样式
	}
}
```

3.animation-duration:动画持续时间

4.animation-delay:动画延迟

5.animation-timing-function:linner

动画执行速度

6.animation-iteration-count:动画执行次数

7.animation-direction:alternate//进行往返动画

8.animation-play-state​:running,paused动画是否进行或者暂停

9.动画的另一种定义方式,设置动画进程的百分比,进行不同的动画

```
@keyframes lnj{
	0%{
	}
	1%{
	}
	2%{
        
	}
}
```

10.animation-fill-mode:指定等待状态和结束状态的样式

backwords:等待状态显示第一帧

forwords:结束状态保持最后一帧的状态

opacity设置为0.5的时候会展示父元素的背景颜色

 ```hyml
ul:hover li{
	opacity: 0.5;
}
ul li:hover{
	opacity: 1;
}
 ```

<h2 id=“4”>3D转换模块</h2>

1.声明3D模块

在父元素中添加transform-style:preserve-3d属性，可以使子元素变为3d模块

2.编写正方体

2.1先写上面，再写后面，再写下面，最后写前面

```html
ul li:nth-child(1){//上
    background-color: red;
    transform: rotateX(90deg) translateZ(100px);
}
ul li:nth-child(2){//后
    background-color: green;
    transform: rotateX(180deg) translateZ(100px);
}
ul li:nth-child(3){//下
    background-color: yellow;
    transform: rotateX(270deg) translateZ(100px);
}
ul li:nth-child(4){//前
    background-color: pink;
    transform: rotateX(360deg) translateZ(100px);
}
ul li:nth-child(5){//右
    background-color: blue;
	transform: translateX(-100px) rotateY(90deg);
}
ul li:nth-child(6){//左
	background-color: skyblue;
	transform:translateX(100px) rotateY(90deg);
}
```

3.编写长方体

只需要编写好正方体之后，添加上transform:scale()；属性就可以转化成长方体

4.保证背景图片不受浏览器宽高的限制

background-size:cover为背景图片的父元素设置这个值即可

5.绝对定位流设置水平居中

left:50%

margin-left:元素宽度的一半

6.overflow：hidden可以隐藏超出屏幕外的动画效果

<h2>背景图片</h2>

1.背景图片大小

background-size:auto auto设置为高度以及宽度的等比拉伸

background-size:cover图片需要等比拉伸，图片需要拉伸到高度以及宽度都填满整个父元素

background-size:content只要宽度或者高度填满之后停止等比拉伸

2.背景图片定位区域属性

background-origin:padding-box从padding区域开始显示

border-box从border区域开始显示

content-box从content区域开始显示

3.背景绘制区域

background-click:控制背景绘制的区域

4.多重背景

编写多重背景的时候要拆开来编写，background-positation控制背景图片的位置

<h2>CSS书写位置</h2>

1.行内样式，直接写到开始标签中

2.内嵌模式，在\<header>\</header>中

3.外链样式 .css文件\<link rel="stylesheet" href="+文件">

4.导入样式

```html
<style>
@import + 文件
</style>
```

企业开发中一般使用外链样式，

1.外链通过link标签来关联，导入样式有兼容问题

2.外链样式会先加载样式，后加载内容，

导入样式先加载内容，后加载样式





