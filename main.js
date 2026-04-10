async function run() {
  const sg = await gulls.init(),
        frag = await gulls.import('./frag.wgsl'),
        compute = await gulls.import('./compute.wgsl'),
        render = gulls.constants.vertex + frag,
        w = window.innerWidth,
        h = window.innerHeight,
        state = new Float32Array(w * h * 2)

  for(let y = 0; y < h; y++) {
    for(let x = 0; x < w; x++) {
      const i = (y * w + x) * 2
      state[i] = 1.0
      const dx = x - w / 2, dy = y - h / 2
      if(Math.abs(dx) < 20 && Math.abs(dy) < 20) {
        state[i + 1] = 1.0
      } else {
        state[i + 1] = 0.0
      }
    }
  }

  const statebuffer1 = sg.buffer(state)
  const statebuffer2 = sg.buffer(state)

  const res = sg.uniform([w, h])
  const feed = sg.uniform(0.041)
  const kill = sg.uniform(0.062)
  const dA = sg.uniform(1.0)
  const dB = sg.uniform(0.5)

  const renderPass = await sg.render({
    shader: render,
    data: [res, sg.pingpong(statebuffer1, statebuffer2)]
  })

  const computePass = sg.compute({
    shader: compute,
    data: [res, feed, kill, dA, dB, sg.pingpong(statebuffer1, statebuffer2)],
    dispatchCount: [Math.round(gulls.width / 8), Math.round(gulls.height / 8), 1],
    times: 25,
  })

  sg.run(computePass, renderPass)

  document.querySelector('#feed').oninput = e => feed.value = parseFloat(e.target.value)
  document.querySelector('#kill').oninput = e => kill.value = parseFloat(e.target.value)
  document.querySelector('#dA').oninput = e => dA.value = parseFloat(e.target.value)
  document.querySelector('#dB').oninput = e => dB.value = parseFloat(e.target.value)
}

run()