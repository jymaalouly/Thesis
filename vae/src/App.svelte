<script>
    import * as tf from "@tensorflow/tfjs";
    import * as d3 from "d3";
    import { onMount } from "svelte";

    class SamplingLayer extends tf.layers.Layer {
        constructor() {
            super({})
        }

        call(inputs) {
            ///console.log(inputs)
            let [z_mean, z_log_sigma] = inputs;
            ///console.log(z_mean, z_log_sigma)
            let res = tf.add(z_mean, tf.mul(tf.exp(z_log_sigma), tf.randomNormal(z_mean.shape, 0, .1)));
            ///console.log("res", res)
            return res
        }

        static get className() {
            return "SamplingLayer"
        }

        computeOutputShape(inputShape) {
            return [3, 7]
        }
    }

    tf.serialization.registerClass(SamplingLayer)





    const rand = d3.randomNormal(.5, .25);
    const width = 128;
    const height = 128;
    const margin = 20;
    let canvas;
    let decoder_canvas;

    let encoder;
    let decoder;

    let n = 50;
    $: data = d3.range(0, n).map(_ => [rand(), rand()]);
    $: x = d3.scaleLinear(d3.extent(data, d => d[0]), [margin, width - margin]);
    $: y = d3.scaleLinear(d3.extent(data, d => d[1]), [width - margin, margin]);
    
    let _img
    onMount(async () => {
        await new Promise(res => {
            let newImg = new Image;
            newImg.onload = function() {
                _img.src = this.src;
                res(_img)
            }
            newImg.src = './bg.png'
        })
        /* const surface = { name: 'Model Summary', tab: 'Model Inspection'};
        const surface2 = {name: "model sum", tab: "qua"}
        tfvis.show.modelSummary(surface, encoder);
        tfvis.show.modelSummary(surface2, decoder); */
    
        encoder = await tf.loadLayersModel("./encoder/model.json")
        decoder = await tf.loadLayersModel("./decoder/model.json")
        draw();
        latent_changed();
        /* const img = tf.browser.fromPixels(canvas, 3).cast('float32').expandDims()//.reshape([1, 128, 128, 3]);
        let res = encoder.predictOnBatch(img)
        latent = await res[2].data() */
    })

    let shapes = d3.symbols
    let marker = 1;
    let size = 5;
    let symbolNames = "circle cross diamond square star triangle wye".split(/ /);
    let colors = ['red','green','blue','cyan','magenta','yellow','black']
    let color = "black"
    $: console.log(marker, size)

    function draw() {
        let context = canvas.getContext("2d");
        context.clearRect(0, 0, width, height)
        context.drawImage(_img, 0, 0);
        data.forEach((d, i) => {
            context.beginPath()
            context.translate(x(d[0]), y(d[1]));
            console.log(marker, d3.symbol().size(size).type(shapes[marker])())
            let p = new Path2D(d3.symbol().size(size * 2).type(shapes[marker])());
            context.fillStyle = color;
            context.fill(p)
            context.setTransform(1, 0, 0, 1, 0, 0);
        })

        const img = tf.browser.fromPixels(canvas, 3).cast('float32').expandDims()//.reshape([1, 128, 128, 3]);
        let res = encoder.predictOnBatch(img)
        latent = res[2].dataSync()
        console.log("latent", latent)
        latent_changed();
    }

    let latent = new Array(7).fill(0)

    function latent_changed() {
        const context = decoder_canvas.getContext("2d");
        context.clearRect(0, 0, width, height)
        let img = decoder.predict(tf.tensor1d(latent).reshape([1, 7]))
        img = img.reshape([128, 128, 3])
        tf.browser.toPixels(img, decoder_canvas);
    }

    async function happy() {
        const data = {};
        let l = new Array(7).fill(0).map(_ => (Math.random() - .5) * 3);
        let img = decoder.predict(tf.tensor1d(l).reshape([1, 7]))
        //console.log(img)
        data.img = img.reshape([128, 128, 3])
        data.latent = l;
        return data;
    }

    function drawHappy(canvas, data) {
        tf.browser.toPixels(data.img, canvas);
    }
</script>

<main>
<img style="visibility: hidden;" bind:this={_img} />
    <form on:change={() => draw()} id="settings">
        <fieldset>
            <legend>data</legend>
            <input type="range" min=10 max=1000 id="size" bind:value={n}><label for="size">number of points: {n}</label><br />
        </fieldset>
        <fieldset>
            <legend>marker</legend>
            <input type="range" min=1 max=20 id="size" bind:value={size}><label for="size">size: {size}</label>
            <br>
            {#each symbolNames as s, i}
                <input type="radio" bind:group={marker} value={i} id={"marker"+i}><label for={"marker"+i} >{s}</label>
            {/each}
            <br>
            {#each colors as c, i}
                <input type="radio" bind:group={color} value={c} id={"color"+i}><label for={"color"+i} style="color: {c};">{c}</label>
            {/each}
        </fieldset>
    </form> 
    <div style="text-align: center;">
        <h2>Input</h2>
        <canvas bind:this={canvas} width={width} height={height} />
    </div>
    <form on:change={() => latent_changed()}>
        <div id="latent" style="display: flex; flex-shrink: true">
            {#each latent as l, i}
                <div><label for={"latent" + i}>{i}:</label> <input id={"latent" + i} type="number" step=.1 bind:value={latent[i]}></div>
            {/each}
        </div>
    </form>
    <div style="text-align: center;">
        <canvas bind:this={decoder_canvas} width={width} height={height} />
        <h2>Output</h2>
    </div>
    <!-- {#if decoder != null}
    <div style="display: flex; flex-wrap: wrap;">
        {#each d3.range(0, 50) as k}
            {#await happy()}
                <span>nope</span>
            {:then data}
            <div>
                <canvas use:drawHappy={data} width={width} height={height}>
                
                </canvas>
                <div id="latentLabel">
                {#each data.latent as l}
                    <span>{d3.format(".1f")(l)}|</span>
                {/each}
                </div>
            </div>
            {/await}
        {/each}
    
    </div>
    {/if} -->
</main>

<style>
    :global(body) {
        font-family: Inter;
    }

    main {
        max-width: 100%;
    }

	label {
        display: inline-block;
        margin: .4rem;
        margin-right: .7rem;
    }

    canvas {
        border: 1px solid grey;
    }

    #latent > div {
        margin: .4rem;
    }

    #latentLabel {
    }

    #latentLabel > span {
        font-size: .4rem;
        font-family: monospace;
    }
</style>