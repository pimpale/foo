import React from 'react';

// Object displaying some image data

interface ImageDataDisplayProps {
  data: ImageData,
  className?: string,
  style?: React.CSSProperties
}

class ImageDataDisplay extends React.Component<ImageDataDisplayProps> {

  private displayCanvas = React.createRef<HTMLCanvasElement>();

  constructor(props: ImageDataDisplayProps) {
    super(props);
    this.paint = this.paint.bind(this);
  }

  componentDidMount() {
    this.paint();
  }

  componentDidUpdate() {
    this.paint();
  }

  paint() {
    const canvas = this.displayCanvas.current;
    if (canvas != null) {
      const context = canvas.getContext("2d");
      if (context != null) {
        context.putImageData(this.props.data, 0, 0);
      }
    }
  }

  render() {
    const { width, height } = this.props.data;
    return <canvas
      className={this.props.className}
      style={this.props.style}
      ref={this.displayCanvas}
      width={width}
      height={height}
    />
  }
}

export default ImageDataDisplay;
