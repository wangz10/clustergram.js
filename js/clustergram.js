// util functions
function getAllLeafs(node) { // recursively find all the leafs for a given node
  if (!node.hasOwnProperty('children')) {
    return [node.id]
  } 
  else {
    if ( !node.children[0].hasOwnProperty('children') && !node.children[1].hasOwnProperty('children') ) {
      return [node.children[0].id, node.children[1].id]
    } else { 
      return getAllLeafs(node.children[0]).concat(getAllLeafs(node.children[1]))
    }
  };
};

function selectText(id) {
    var doc = document;
    var text = doc.getElementById(id);    
    if (doc.body.createTextRange) { // ms
        var range = doc.body.createTextRange();
        range.moveToElementText(text);
        range.select();
    } else if (window.getSelection) { // moz, opera, webkit
        var selection = window.getSelection();            
        var range = doc.createRange();
        range.selectNodeContents(text);
        selection.removeAllRanges();
        selection.addRange(range);
    }
};

function findMax(mat) { //find max number from a matrix
  var max = 0
  for (var i = 0; i < mat.length; i++) {
    var row = mat[i];
    var rowMax = Math.max.apply(Math, row);
    if (rowMax > max) {
      max = rowMax;
    };
  };
  return max
};

function findMin(mat) {
  var min = 0
  for (var i = 0; i < mat.length; i++) {
    var row = mat[i];
    var rowMin = Math.min.apply(Math, row);
    if (rowMin < min) {
      min = rowMin;
    };
  };
  return min
};

// layout parameters
var widthH = 500, // width of heatmap
    heightH = 500,
    widthR = 300, // width of row dendrogram
    heightR = heightH,
    margin = 30,
    width = margin*3 + widthH + widthR,
    height = margin*2 + heightH;

// for row dendrogram
var cluster = d3.layout.cluster()
    .size([heightR, widthR])
    .separation(function(a, b){
      return a.parent == b.parent ? 1 : 1; // do not discern separation of non sister nodes
    })

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
var zoomLayer = svg.append('g')
    .attr('id','zoomLayer')

  rowDendroContainer = zoomLayer.append("g")
    // .attr("transform", "translate(40,0)");

var genes = 0;

d3.json("json/testRand_dendroRow.json", function(error, root) { // loading row dendrogram data
  // console.log(root);
  genes = getAllLeafs(root) // array of genes in the layout order
  // console.log(genes)
  var rowDendroScale = d3.scale.linear()
      .domain([0, root.height])
      // .range([width/2, width/4])
      .range([widthR-margin,margin])

  var nodes = cluster.nodes(root).map(function(d) {d.y = rowDendroScale(d.height); return d});
  var links = cluster.links(nodes);

  var link = rowDendroContainer.selectAll(".link")
      .data(links)
    .enter().append("path")
      .attr("class", "link")
      .attr("d", diagonal)

  var node = rowDendroContainer.selectAll(".node")
      .data(nodes)
    .enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; })
      // .attr('id', function(d) {return d.id})

  node.append("circle")
      .attr("r", function(d){
        return d.children ? heightH/genes.length : heightH/genes.length/2;
      })
      .attr('id', function(d) {return 'circle'+d.id})
      .on('mouseover', function(d){
        var allLeafs = getAllLeafs(d)
        for (var i = 0; i < allLeafs.length; i++) {
          d3.select('#text' + allLeafs[i]).style('fill','red')
          d3.select('#circle' + allLeafs[i]).style('stroke','red').style('fill','red')
        };
      })
      .on('mouseout', function(d){
        d3.selectAll('text').style('fill', 'black')
        d3.selectAll('circle').style('stroke','steelblue').style('fill','steelblue')
      })
      .on('click', function(d){ // click
        var allLeafs = getAllLeafs(d)
        allLeafs = allLeafs.join('\n')
        d3.select('#genesTextarea').remove()
        var textarea = d3.select('body').append('textarea')
          .attr('id','genesTextarea')
          .attr('cols', 10)
          .attr('rows', 10)
          .text(allLeafs)
      })

  node.append("text")
      .attr("dx", function(d) { return d.children ? -8 : 8; })
      .attr("dy", 2)
      .attr('id', function(d) {return 'text'+d.id })
      .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
      .style("font-size", heightH/genes.length+'px')
      .text(function(d) { return d.children ? this.remove() : d.id; });
});


// for the heatmap
d3.json('json/testRand_matrix.json',function(mat){
  var minData = findMin(mat),
    maxData = findMax(mat),
    numRows = mat.length,
    numCols = mat[0].length;

  var colorScale = d3.scale.linear()
    .domain([minData, 0, maxData])
    .range(['blue','white','red'])
    // .range(['#1f77b4','#fff','#d62728']);
  var xScale = d3.scale.linear()
    .domain([0, numCols])
    .range([0, widthH])
  var yScale = d3.scale.linear()
    .domain([0, numRows])
    .range([0, heightH])

  var zoom = d3.behavior.zoom()
    .scaleExtent([1, 100])
    .on("zoom", zoomed)

  svg.call(zoom)

  var heatmapContainer = zoomLayer.append('g')
    .attr('transform',function(){
      return 'translate('+widthR +',0)';
    })

  var heatmapRows = heatmapContainer.selectAll('g').data(mat)
    .enter()
    .append('g')
    .attr('transform',function(d,i){
      return 'translate(0,'+yScale(i) +')';
    })
    .on('mouseover', function(d, i){
      d3.select('body').append('div').attr('id','title')
        .append('span')
        .text('gene:'+genes[i])
        .append('br')
      d3.select('#title').append('span').text('vals:'+d)
      d3.select('#text' + genes[i]).style('fill','red')
    })
    .on('mouseout', function(d, i){
      d3.select('#title').remove()
      d3.select('#text' + genes[i]).style('fill','black')
    })

  var heatmaptCells = heatmapRows.selectAll('rect')
    .data(function(row){return row;})
    .enter()
    .append('svg:rect')
    .attr('width', widthH/numCols)
    .attr('height', heightH/numRows)
    .attr('x',function(d, i){
      return xScale(i)
    })
    .style('fill',function(d){
      return colorScale(d);
    })
    .on('mouseover', function(d, i){
      var sampleTitle = d3.select(this).append('title').attr('id','sampleTitle')
      sampleTitle.append('div').text('sample'+i)
      // sampleTitle.append('br')
      sampleTitle.append('div').text(', value: '+ Math.round(d*1000)/1000)
      // d3.select('#title').append('span').text('sample'+i)
    })
    .on('mouseout', function(d, i){
      d3.select('#sampleTitle').remove()
    })


function zoomed() {
  zoomLayer.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
}

})



// d3.select(self.frameElement).style("height", height + "px");

	