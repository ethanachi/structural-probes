var USE_MULTILINGUAL = true;

const LANGS = ["ar", "de", "es", "en", "fa", "fi", "fr", "id", "zh"]
const VISUALIZATION_SIZE = 0.7;
const VIEW_SIZE = 1.3;

var margin = {top: 20, right: 20, bottom: 20, left: 40},
    width = window.innerHeight * .8
    height = window.innerHeight * .8

/*
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */

// setup x
var xValue = function(d) { return d.x0;}, // data -> value
    xScale = d3.scaleLinear().domain([-65, 65]).range([0, width]), // value -> display
    xMap = function(d) { return xScale(xValue(d));}, // data -> display
    xAxis = d3.axisBottom(xScale);

// setup y
var yValue = function(d) { return d.x1;}, // data -> value
    yScale = d3.scaleLinear().domain([-65, 65]).range([height, 0]), // value -> display
    yMap = function(d) { return yScale(yValue(d));}, // data -> display
    yAxis = d3.axisLeft(yScale);

// setup fill color
//var cValue = function(d) {return d.label;},
//    color = d3.scaleOrdinal(d3.schemeCategory10);

colorMap = {
    "det": "#338833",
    "amod": "#8f0909",
    "nsubj": "#dbb02e",
    "case": "#152894",
    "conj": "#ff8400",
    "cc": "#696969",
    "advmod": "#d95cc6",
    "cop": "#1ab7b7",
   "xcomp": "#39ce78",
   "obl": "#ba95d1",
   "xcomp": "#39ce78",
   "obj": "#853ab4",
    "expl": "#6de5c0"

    //    "fr-cop": "#d95cc6",
    //    "en-cop": "#91218d",
    //    "fr-xcomp": "#12776c",
    //    "en-xcomp": "#25e5dd",
    //    "fr-obl": "#5e910b",
    //    "en-obl": "#cbde36",
    //    "fr-nmod": "#ba1273",
    //    "en-nmod": "#831b57",
}

//colorMap = {
//    "zh-det": "#338833",
//    "en-det": "#88ee88",
//    "zh-amod": "#8f0909",
//    "en-amod": "#db5e35",
//    "zh-nsubj": "#dbb02e",
//    "en-nsubj": "#a65b11",
//    "zh-case": "#152894",
//    "en-case": "#5f72de",
//    "zh-conj": "#ff8400",
//    "en-conj": "#964f02",
//    "zh-cc": "#696969",
//    "en-cc": "#bdbdbd",
//    "zh-advmod": "#a25700",
//    "en-advmod": "#ff7600",
//    "zh-obl": "#d95cc6",
//    "en-obl": "#91218d",
//}

// add the graph canvas to the body of the webpage
var svg = d3.select("#svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
.attr("class", "sentenceTooltip")
.style("opacity", 0);

var dataField = d3.select(".field");

svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis)
    .append("text")
    .attr("class", "label")
    .attr("x", width)
    .attr("y", -6)
    .style("text-anchor", "end")
    .text("x0");

// y-axis
svg.append("g")
    .attr("class", "y axis")
    .call(yAxis)
    .append("text")
    .attr("class", "label")
    .attr("transform", "rotate(-90)")
    .attr("y", 6)
    .attr("dy", ".71em")
    .style("text-anchor", "end")
    .text("x1");

console.log('svg loaded');
// load data
function loadData(langs) {
    svg.selectAll(".dot").remove();
    // loader settings
    var opts = {
      lines: 9, // The number of lines to draw
      length: 9, // The length of each line
      width: 5, // The line thickness
      radius: 14, // The radius of the inner circle
      color: '#EE3124', // #rgb or #rrggbb or array of colors
      speed: 1.9, // Rounds per second
      trail: 40, // Afterglow percentage
      className: 'spinner', // The CSS class to assign to the spinner
    };
    var target = document.getElementById("visualization");
    console.log(target);
    var spinner = new Spinner(opts).spin(target);
    var spinnerElem = document.getElementsByClassName('spinner')[0];
    spinnerElem.style.left = "25%";
    var loading = document.getElementById("loading");
    loading.style.opacity = 1.0;
    d3.tsv("all-new.tsv", (d) => {
        let idx = parseInt(d.idx);
        let newHTML = d.sentence.split(' ');
        newHTML.splice(idx, 0, "<b>");
        newHTML.splice(idx+2, 0, "</b>");
        newHTML = newHTML.join(" ");
        newHTML += "<br/>" + d.label;
        newHTML += `<br/> (${+d.x0} ${+d.x1})`;
        if (d.label.includes(":")) d.label = d.label.substr(0, d.label.indexOf(':'));
        return {
            x0: +d.x0,
            x1: +d.x1,
            label: USE_MULTILINGUAL ? d.label.split('-')[1] : d.label,
            sentence: d.sentence,
            idx: idx,
            html: newHTML,
            dot_id: "dot-" + d.x0.replace('.', '_'),
            lang: d.label.split('-')[0]
        }
    }).then((data) => {

        data = data.filter((d) => {
            return (langs.includes(d.lang) && colorMap.hasOwnProperty(d.label));
        });
        d3.shuffle(data);


        svg.selectAll(".dot")
            .data(data)
            .enter().append("circle")
            .attr("class", "dot")
            .style("fill", "purple")
            .attr("r", VIEW_SIZE)
            .style("fill", function(d) { return colorMap[d.label];})
            .attr("cx", xMap)
            .attr("cy", yMap)
            .attr("id", function(d) { return d.dot_id; })
            .on("mouseover", function(d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", 1);
                tooltip.html(d.html)
                    .style("left", (d3.event.pageX + 5) + "px")
                    .style("top", (d3.event.pageY - 28) + "px");
            })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            })
            .on("click", function(d) {
                svg.select(".selectedDot")
                    .attr("r", 1)
                    .classed("selectedDot", false);
                svg.select("#" + d.dot_id)
                    .attr("r", 10)
                    .classed("selectedDot", true);
                dataField.html(d.html);
            });
        spinner.stop();
        loading.style.opacity = 0;
    });
}

const checkboxes = document.getElementById('language-checkboxes');
for (const lang of LANGS) {
    const newCheckbox = `<input type="checkbox" id="check-${lang}" class="checkbox" checked> `;
    checkboxes.insertAdjacentHTML('beforeend', newCheckbox);
    const newLabel = `<label class="form-check-label" for="exampleCheck1">${lang}</label><br/>`;
    checkboxes.insertAdjacentHTML('beforeend', newLabel);
    console.log(newCheckbox);
}

const legend = document.getElementById('legend');
for (const label in colorMap) {
    if (colorMap.hasOwnProperty(label)) {
        const line = `<p class="legendRow"><span class="dot" style="background-color:${colorMap[label]}"></span> ${label}`;
        console.log(line);
        legend.insertAdjacentHTML('beforeend', line);
    }
}
// <span class="dot"></span> xcomp

loadData(LANGS);



function reloadData() {
    loadData(LANGS.filter((lang) => {
        return document.getElementById(`check-${lang}`).checked;
    }));
}

function selectAll() {
    LANGS.forEach((lang) => {
        document.getElementById(`check-${lang}`).checked = true;
    });
}

function saveImage() {
    //get svg element.
    var svg = document.getElementById("svg");

    //get svg source.
    var serializer = new XMLSerializer();
    var source = serializer.serializeToString(svg);

    //add name spaces.
    if(!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)){
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    if(!source.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)){
        source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    }

    //add xml declaration
    source = '<?xml version="1.0" standalone="no"?>\r\n' + source;

    //convert svg source to URI data scheme.
    var url = "data:image/svg+xml;charset=utf-8,"+encodeURIComponent(source);

    //set url value to a element's href attribute.
    document.getElementById("link").href = url;
    //you can download svg file by right click menu.
}
