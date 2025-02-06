import React from 'react';
import ScalarMap from '../utils/ScalarMap';
import VectorMap from '../utils/VectorMap';


type VectorMapDisplayParticle = {
  x: number;
  y: number;
  age: number;
}

interface VectorMapDisplayState {
  alphaData: ScalarMap;
  particles: VectorMapDisplayParticle[];
  intervalCode: number | null;
}

interface VectorMapDisplayProps {
  vmap: VectorMap;
  base: ImageData;
  className?: string;
  style?: React.CSSProperties;
}

class VectorMapDisplay extends React.Component<VectorMapDisplayProps, VectorMapDisplayState> {
  private displayCanvas = React.createRef<HTMLCanvasElement>();

  constructor(props: VectorMapDisplayProps) {
    super(props);
    this.paint = this.paint.bind(this);
    this.tick = this.tick.bind(this);
    const { width, height } = props.base;
    this.state = {
      alphaData: new ScalarMap(width, height),
      particles: new Array<VectorMapDisplayParticle>(),
      intervalCode: null,
    }
  }

  componentDidMount() {
    this.setState({ intervalCode: window.setInterval(this.tick, 100) });
  }

  componentWillUnmount() {
    if (!!this.state.intervalCode) {
      window.clearInterval(this.state.intervalCode)
    }
    this.setState({ intervalCode: null })
  }

  componentDidUpdate() {
    this.paint();
  }


  tick() {
    const { width, height } = this.state.alphaData.dims();
    // update particles
    const nparticles = this.state.particles.map(
      (p) => {
        // move particles
        const dir = this.props.vmap.getv(Math.floor(p.x), Math.floor(p.y));
        return {
          x: p.x + dir[0] / 2,
          y: p.y + dir[1] / 2,
          age: p.age + 1
        }
      })
      // decay particles that are outside boundaries or too old
      .filter((p) => p.x > 0 && p.x < width && p.y > 0 && p.y < height && p.age < 100)
      .concat(
        // Add more particles
        [...Array<VectorMapDisplayParticle>(4)]
          .map(_ => ({
            x: Math.random() * width,
            y: Math.random() * height,
            age: 0,
          }))
      );

    // regen imagedata
    let ndata = new ScalarMap(width, height);
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        ndata.setv(x, y, this.state.alphaData.getv(x, y) / 1.1);
      }
    }

    for (const p of nparticles) {
      ndata.setv(Math.floor(p.x), Math.floor(p.y), 1);
    }


    this.setState({
      alphaData: ndata,
      particles: nparticles
    });
  }

  paintHeightMap(ctx: CanvasRenderingContext2D) {
    const { width, height } = this.props.base;
    ctx.putImageData(this.props.base, 0, 0);
    let final = new ImageData(width, height);
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const bindex = (width * y + x) * 4;
        final.data[bindex + 0] = this.props.base.data[bindex + 0];
        final.data[bindex + 1] = this.props.base.data[bindex + 1] + this.state.alphaData.getv(x, y) * 255;
        final.data[bindex + 2] = this.props.base.data[bindex + 2];
        final.data[bindex + 3] = this.props.base.data[bindex + 3];
      }
    }
    ctx.putImageData(final, 0, 0);
  }

  paint() {
    const maybeCanvas = this.displayCanvas.current;
    if (maybeCanvas != null) {
      const maybeContext = maybeCanvas.getContext("2d");
      if (maybeContext != null) {
        this.paintHeightMap(maybeContext);
      }
    }
  }

  render() {
    const { width, height } = this.props.base;
    return (
      <canvas
        style={this.props.style}
        className={this.props.className}
        ref={this.displayCanvas}
        width={width}
        height={height}
      />
    );
  }
}

export default VectorMapDisplay;
