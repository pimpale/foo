import React from 'react';
import { crop } from '../utils/image';
import { clamp } from '../utils/math';

import { Arrow90degDown, Arrow90degUp } from 'react-bootstrap-icons';

// Object displaying some image data


interface ImageDataDisplayProps {
  data: ImageData,
  zoomRadius: number,
  displayHeight: number,
  className?: string,
  style?: React.CSSProperties,
  showInstructions?: boolean
}

class ImageDataDisplay extends React.Component<ImageDataDisplayProps> {

  private displayCanvas = React.createRef<HTMLCanvasElement>();
  private closeupCanvas = React.createRef<HTMLCanvasElement>();

  private mloc: { x: number, y: number } | null = null;

  constructor(props: ImageDataDisplayProps) {
    super(props);
    this.paint = this.paint.bind(this);
  }

  componentDidMount() {
    this.displayCanvas.current!.addEventListener('mousemove', this.handleMouseOver)
    this.displayCanvas.current!.addEventListener('mouseleave', this.handleMouseLeave)

    this.paint();
  }

  getMousePos = (evt: MouseEvent) => {
    const canvas = this.displayCanvas.current!;
    const rect = canvas.getBoundingClientRect(); // abs. size of element
    const scaleX = canvas.width / rect.width;    // relationship bitmap vs. element for X
    const scaleY = canvas.height / rect.height;  // relationship bitmap vs. element for Y

    return {
      x: (evt.clientX - rect.left) * scaleX,   // scale mouse coordinates after they have
      y: (evt.clientY - rect.top) * scaleY     // been adjusted to be relative to element
    }
  }

  handleMouseOver = (evt: MouseEvent) => {
    this.mloc = this.getMousePos(evt);
    this.paint();
  }

  handleMouseLeave = () => {
    this.mloc = null;
    this.paint();
  }

  componentDidUpdate() {
    this.paint();
  }

  paint() {
    const displayCanvas = this.displayCanvas.current!;
    const displayContext = displayCanvas.getContext("2d");

    const closeupCanvas = this.closeupCanvas.current!;

    if (displayContext != null) {
      displayContext.putImageData(this.props.data, 0, 0);

      // if we've been mouse overed
      if (this.mloc != null) {
        // find x and y coords
        const zr = this.props.zoomRadius;
        const x = Math.floor(clamp(this.mloc.x - zr / 2, zr, this.props.data.width - zr));
        const y = Math.floor(clamp(this.mloc.y - zr / 2, zr, this.props.data.height - zr));

        // draw on closeup canvas
        const closeupContext = closeupCanvas.getContext("2d");
        if (closeupContext != null) {
          closeupContext.imageSmoothingEnabled = false;
          const croppedData = crop(this.props.data, x, y, this.props.zoomRadius, this.props.zoomRadius);
          closeupContext.putImageData(croppedData, 0, 0);
        }

        // draw rectangle on bigger canvas
        displayContext.fillStyle = "#FF0000";
        displayContext.fillRect(x, y, this.props.zoomRadius, this.props.zoomRadius)
      } else {
        // if null then clear canvas
        const closeupContext = closeupCanvas.getContext("2d");
        if (closeupContext != null) {
          closeupContext.clearRect(0, 0, this.props.zoomRadius, this.props.zoomRadius)
        }
      }
    }
  }

  render() {
    const { width, height } = this.props.data;
    return <div style={this.props.style} className={this.props.className}>
      <div className='text-center pb-3' hidden={!this.props.showInstructions}>
        <Arrow90degDown className='fs-3' style={{ transform: "translateY(0.5rem)" }} />
        <span className='fs-5' style={{ fontFamily: "Permanent Marker" }}> Hover over me!</span>
      </div>
      <div className="d-flex align-items-center justify-content-center">
        <canvas
          className="border border-dark mx-2"
          ref={this.displayCanvas}
          width={width}
          height={height}
        />
        <canvas
          className="border border-dark mx-2"
          style={{
            height: this.props.displayHeight / 2,
            width: this.props.displayHeight / 2,
          }}
          ref={this.closeupCanvas}
          width={this.props.zoomRadius}
          height={this.props.zoomRadius}
        />
      </div>
    </div>
  }
}

export default ImageDataDisplay;
