var USE_MULTILINGUAL = false;

var margin = {top: 20, right: 20, bottom: 20, left: 40},
    width = 700 - margin.left - margin.right,
    height = 700 - margin.top - margin.bottom;

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
    "fr-det": "#338833",
    "en-det": "#88ee88",
    "fr-amod": "#8f0909",
    "en-amod": "#db5e35",
    "fr-nsubj": "#dbb02e",
    "en-nsubj": "#a65b11",
    "fr-case": "#152894",
    "en-case": "#5f72de",
    "fr-conj": "#ff8400",
    "en-conj": "#964f02",
    "fr-cc": "#696969",
    "en-cc": "#bdbdbd",
    "fr-advmod": "#d95cc6",
    "en-advmod": "#91218d",
    "det": "#338833",
    "amod": "#8f0909",
    "nsubj": "#dbb02e",
    "case": "#152894",
    "conj": "#ff8400",
    "cc": "#696969",
    "advmod": "#d95cc6",
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
//    "fi-det": "#338833",
//    "en-det": "#88ee88",
//    "fi-amod": "#8f0909",
//    "en-amod": "#db5e35",
//    "fi-nsubj": "#dbb02e",
//    "en-nsubj": "#a65b11",
//    "fi-case": "#152894",
//    "en-case": "#5f72de",
//    "fi-conj": "#ff8400",
//    "en-conj": "#964f02",
//    "fi-cc": "#696969",
//    "en-cc": "#bdbdbd",
//    "fi-advmod": "#a25700",
//    "en-advmod": "#ff7600",
//    "fi-obl": "#d95cc6",
//    "en-obl": "#91218d",
//}

// add the graph canvas to the body of the webpage
var svg = d3.select("body").append("svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
.attr("class", "tooltip")
.style("opacity", 0);

// load data
d3.tsv("data.tsv", (d) => {
    console.log(d);
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
        html: newHTML
    }
}).then((data) => {

    data = data.filter((d) => {
        return (colorMap.hasOwnProperty(d.label));
    });
    d3.shuffle(data);
    console.log(data);

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

    svg.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .style("fill", "purple")
        .attr("r", 1)
        .style("fill", function(d) { return colorMap[d.label];})
        .attr("cx", xMap)
        .attr("cy", yMap)
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

    });
});



//       function(error, data) {
//  console.log(data);
//  // change string (from CSV) into number format
//  data.forEach(function(d) {
//    d.x0 = +d.x0;
//    d.x1 = +d.x1;
////    console.log(d);
//  });
//
//  // don't want dots overlapping axis, so add in buffer to data domain
//  xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
//  yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);
//
//  // x-axis


//  // draw dots

//  // draw legend
//  var legend = svg.selectAll(".legend")
//      .data(color.domain())
//    .enter().append("g")
//      .attr("class", "legend")
//      .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });
//
//  // draw legend colored rectangles
//  legend.append("rect")
//      .attr("x", width - 18)
//      .attr("width", 18)
//      .attr("height", 18)
//      .style("fill", color);
//
//  // draw legend text
//  legend.append("text")
//      .attr("x", width - 24)
//      .attr("y", 9)
//      .attr("dy", ".35em")
//      .style("text-anchor", "end")
//      .text(function(d) { return d;})
//});