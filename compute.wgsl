@group(0) @binding(0) var<uniform> res: vec2f;
@group(0) @binding(1) var<uniform> feed: f32;
@group(0) @binding(2) var<uniform> kill: f32;
@group(0) @binding(3) var<uniform> dA: f32;
@group(0) @binding(4) var<uniform> dB: f32;
@group(0) @binding(5) var<storage> statein: array<f32>;
@group(0) @binding(6) var<storage, read_write> stateout: array<f32>;

fn idx(x:i32, y:i32) -> u32 {
  return u32(y * i32(res.x) + x) * 2;
}

fn laplaceA(x:i32, y:i32) -> f32 {
  return statein[idx(x, y)] * -1.0
       + statein[idx(x - 1, y)] * 0.2
       + statein[idx(x + 1, y)] * 0.2
       + statein[idx(x, y + 1)] * 0.2
       + statein[idx(x, y - 1)] * 0.2
       + statein[idx(x - 1, y - 1)] * 0.05
       + statein[idx(x + 1, y - 1)] * 0.05
       + statein[idx(x + 1, y + 1)] * 0.05
       + statein[idx(x - 1, y + 1)] * 0.05;
}

fn laplaceB(x:i32, y:i32) -> f32 {
  return statein[idx(x, y) + 1] * -1.0
       + statein[idx(x - 1, y) + 1] * 0.2
       + statein[idx(x + 1, y) + 1] * 0.2
       + statein[idx(x, y + 1) + 1] * 0.2
       + statein[idx(x, y - 1) + 1] * 0.2
       + statein[idx(x - 1, y - 1) + 1] * 0.05
       + statein[idx(x + 1, y - 1) + 1] * 0.05
       + statein[idx(x + 1, y + 1) + 1] * 0.05
       + statein[idx(x - 1, y + 1) + 1] * 0.05;
}

fn styleMap(x:i32, y:i32) -> vec2f {
  let nx = f32(x) / res.x - 0.5;
  let ny = f32(y) / res.y - 0.5;
  let localFeed = clamp(feed + nx * 0.02, 0.01, 0.1);
  let localKill = clamp(kill + ny * 0.008, 0.04, 0.07);
  return vec2f(localFeed, localKill);
}

@compute
@workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) _cell:vec3u) {
  let x = i32(_cell.x);
  let y = i32(_cell.y);
  let w = i32(res.x);
  let h = i32(res.y);
  let i = idx(x, y);

  if(x < 1 || x >= w - 1 || y < 1 || y >= h - 1) {
    stateout[i] = statein[i];
    stateout[i + 1] = statein[i + 1];
    return;
  }

  let a = statein[i];
  let b = statein[i + 1];
  let params = styleMap(x, y);
  let localFeed = params.x;
  let localKill = params.y;

  let newA = a + (dA * laplaceA(x, y)) - (a * b * b) + (localFeed * (1.0 - a));
  let newB = b + (dB * laplaceB(x, y)) + (a * b * b) - ((localKill + localFeed) * b);

  stateout[i] = clamp(newA, 0.0, 1.0);
  stateout[i + 1] = clamp(newB, 0.0, 1.0);
}