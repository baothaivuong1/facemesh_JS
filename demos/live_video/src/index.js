/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import '@tensorflow-models/face-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE, createDetector} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';
import { math, norm } from '@tensorflow/tfjs-core';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

// Distance 2 điểm Oxyz
function distance(a,b) {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2) + Math.pow(a.z - b.z, 2))
}

// Function xác định faceshape
function face_shape(a,b,c,d) {
    if (c > b > a) {
        return 'Triangle'
    }
    else if (a > b) {
        return 'Heart'
    }
    else if (Math.abs(a-b) <= 20 && Math.abs(b-c) <= 20 && d > a && d > b && d > c) {
        return 'Oblong'
    }
    else if (Math.abs(b-d) <= 20 && b > a && b > c) {
        return 'Round'
    }
    else if (d > b > a > c) {
        return 'Diamond'
    }
    else if (d > b && a > c) {
        return 'Oval'
    }
    else {
        return "Undifined"
    }
}


async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateFaceStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateFaceStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let faces = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateFaces.
    beginEstimateFaceStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      faces =
          await detector.estimateFaces(camera.video, {flipHorizontal: false});
      if (faces.length > 0) {
        let lst = faces[0].keypoints
        let left_radius = distance(lst[473], lst[474])
        let ratio = 5.5/ left_radius
        
        // các giá trị tính kính
        let temple_distance_mm = distance(lst[46], lst[276]) * ratio
        let pd_distance_mm = distance(lst[468], lst[473]) * ratio
        let inner_eyetail_distance_mm = distance(lst[243], lst[463]) * ratio
        let outer_eyetail_distance_mm = distance(lst[130], lst[359]) * ratio

        // các giá trị tính face_shape
        let forehead_distance = distance(lst[70], lst[300])
        let cheekbone_distance = distance(lst[111], lst[340])
        let facelength_distance = distance(lst[152], lst[10]) * 0.87
        let jawline_distance = (distance(lst[172], lst[136]) + distance(lst[136], lst[150]) 
        + distance(lst[150], lst[149]) + distance(lst[149], lst[176]) + distance(lst[176], lst[148])
        + distance(lst[148], lst[152]) + distance(lst[152], lst[377]) + distance(lst[377], lst[400])
        + distance(lst[400], lst[378]) + distance(lst[378], lst[379]) + distance(lst[379], lst[365])
        + distance(lst[365], lst[397]))
        
        

        // faceshape
        let shape = face_shape(forehead_distance, cheekbone_distance, jawline_distance, facelength_distance)

        // console ra giá trị
        console.log("temple: " + temple_distance_mm)
        console.log("pd: " + temple_distance_mm)
        console.log("inner_eyetail: " + temple_distance_mm)
        console.log("outer_eyetail: " + temple_distance_mm)
        console.log("Face shape: " + shape)

      }
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateFaceStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (faces && faces.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(
        faces, STATE.modelConfig.triangulateMesh,
        STATE.modelConfig.boundingBox);
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();
};

app();
